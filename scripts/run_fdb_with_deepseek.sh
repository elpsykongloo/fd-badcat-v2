#!/bin/bash
#
# run_fdb_with_deepseek.sh
#
# Wrapper script to run FDB evaluation with DeepSeek v4-flash judge.
# Ensures proxy is disabled and proper environment is loaded.
#
# Usage:
#   ./scripts/run_fdb_with_deepseek.sh --results-dir <dir> [other FDB options]
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FDB_DIR="/root/autodl-tmp/FDBench_v3/v3"
EVAL_ENV="$REPO_ROOT/configs/eval.env"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}FDB Evaluation with DeepSeek v4-flash Judge${NC}"
echo "============================================================"

# 1. Disable proxy for DeepSeek direct connection
echo -e "\n${YELLOW}1. Disabling proxy environment variables...${NC}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# Judge is a pure-API workload — default to high concurrency (llm_judge.py native knob)
export FDB_LLM_WORKERS="${FDB_LLM_WORKERS:-100}"
unset all_proxy ALL_PROXY no_proxy NO_PROXY
echo "   ✓ Proxy disabled (DeepSeek uses direct connection)"

# 2. Load eval environment
echo -e "\n${YELLOW}2. Loading evaluation environment...${NC}"
if [[ -f "$EVAL_ENV" ]]; then
    source "$EVAL_ENV"
    echo "   ✓ Loaded: $EVAL_ENV"
    echo "   Model: $DEEPSEEK_MODEL"
    echo "   API Key: ***${DEEPSEEK_API_KEY: -8}"
else
    echo -e "   ${RED}✗ Not found: $EVAL_ENV${NC}"
    exit 1
fi

# 3. Verify DeepSeek key is loadable
echo -e "\n${YELLOW}3. Verifying DeepSeek key...${NC}"
if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
    echo -e "   ${RED}✗ DEEPSEEK_API_KEY empty after sourcing $EVAL_ENV${NC}"
    exit 1
fi
echo "   ✓ DeepSeek key loaded (model: ${DEEPSEEK_MODEL:-deepseek-v4-flash}, workers: $FDB_LLM_WORKERS)"

# 4. Run FDB evaluation
echo -e "\n${YELLOW}4. Running FDB evaluation...${NC}"
cd "$FDB_DIR"

# Build FDB command with DeepSeek judge configuration
FDB_CMD="python3 evaluate_tool_calls.py \
    --llm-provider deepseek \
    --llm-model $DEEPSEEK_MODEL \
    --llm-base-url $DEEPSEEK_BASE_URL \
    --llm-api-key $DEEPSEEK_API_KEY \
    --llm-timeout $DEEPSEEK_TIMEOUT \
    --use-llm \
    $@"

echo "   Command: $FDB_CMD"
echo ""

# Execute
eval $FDB_CMD

EXIT_CODE=$?

if [[ $EXIT_CODE -eq 0 ]]; then
    echo -e "\n${GREEN}✓ FDB evaluation completed successfully${NC}"
else
    echo -e "\n${RED}✗ FDB evaluation failed (exit code: $EXIT_CODE)${NC}"
fi

exit $EXIT_CODE
