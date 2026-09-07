// Fixed 256-sample float32/16k input packets, independent of device block size.
// Area averaging carries fractional samples across render calls when the audio
// device does not honour the requested 16 kHz AudioContext rate.
class MicFrames extends AudioWorkletProcessor {
  constructor(options = {}) {
    super();
    this.frame = new Float32Array(256);
    this.used = 0;
    this.width = sampleRate / 16000;
    this.remaining = this.width;
    this.sum = 0;
    this.referenceInput = !!options.processorOptions?.inputReference;
    this.refFrame = new Float32Array(256);
    this.refSum = 0;
    this.seq = 0;
  }
  process(inputs) {
    const input = inputs[0]?.[0];
    const reference = inputs[1]?.[0];
    if (input) for (let index = 0; index < input.length; index++) {
      const value = input[index], refValue = reference?.[index] || 0;
      let available = 1;
      while (available > 1e-9) {
        const take = Math.min(available, this.remaining);
        this.sum += value * take;
        this.refSum += refValue * take;
        this.remaining -= take;
        available -= take;
        if (this.remaining < 1e-9) {
          this.frame[this.used] = this.sum / this.width;
          this.refFrame[this.used++] = this.refSum / this.width;
          this.sum = 0;
          this.refSum = 0;
          this.remaining = this.width;
          if (this.used === 256) {
            if (this.referenceInput) {
              const raw = new ArrayBuffer(16 + 256 * 4), view = new DataView(raw);
              view.setUint32(0, 0x314d4446, true); // FDM1
              view.setUint32(4, this.seq++, true);
              view.setUint32(8, 256, true); view.setUint32(12, 16000, true);
              for (let i = 0; i < 256; i++) {
                view.setInt16(16 + i * 4, Math.max(-32768, Math.min(32767, Math.round(this.frame[i] * 32768))), true);
                view.setInt16(18 + i * 4, Math.max(-32768, Math.min(32767, Math.round(this.refFrame[i] * 32768))), true);
              }
              this.port.postMessage(raw, [raw]);
            } else this.port.postMessage(this.frame.buffer, [this.frame.buffer]);
            this.frame = new Float32Array(256);
            this.used = 0;
          }
        }
      }
    }
    return true;
  }
}
registerProcessor("mic-frames", MicFrames);
