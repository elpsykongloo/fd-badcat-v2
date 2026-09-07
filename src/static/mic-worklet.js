// Fixed 256-sample float32/16k input packets, independent of device block size.
// Area averaging carries fractional samples across render calls when the audio
// device does not honour the requested 16 kHz AudioContext rate.
class MicFrames extends AudioWorkletProcessor {
  constructor() {
    super();
    this.frame = new Float32Array(256);
    this.used = 0;
    this.width = sampleRate / 16000;
    this.remaining = this.width;
    this.sum = 0;
  }
  process(inputs) {
    const input = inputs[0]?.[0];
    if (input) for (const value of input) {
      let available = 1;
      while (available > 1e-9) {
        const take = Math.min(available, this.remaining);
        this.sum += value * take;
        this.remaining -= take;
        available -= take;
        if (this.remaining < 1e-9) {
          this.frame[this.used++] = this.sum / this.width;
          this.sum = 0;
          this.remaining = this.width;
          if (this.used === 256) {
            this.port.postMessage(this.frame.buffer, [this.frame.buffer]);
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
