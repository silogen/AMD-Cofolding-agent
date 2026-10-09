#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# Environment template for Drug Discovery Agent on Slurm.
# Copy to env.local.sh and fill in your values.  Both setup.sh and the
# Slurm scripts source this file.
# ──────────────────────────────────────────────────────────────────────

# --- Project paths (edit to match your cluster layout) ----------------
export REINVENT4_PROJECT_DIR="${REINVENT4_PROJECT_DIR:-$HOME/dd_agentic}"
export REINVENT4_ROOT="${REINVENT4_PROJECT_DIR}/REINVENT4"
export REINVENT4_MCP_CONFIG="${REINVENT4_PROJECT_DIR}/reinvent4_mcp_config.yaml"
export REINVENT4_VENV="${REINVENT4_PROJECT_DIR}/.venv"

# --- LLM credentials --------------------------------------------------
# API-management gateway: set LLM_GATEWAY_KEY (sent as
# Ocp-Apim-Subscription-Key).  Any other endpoint (Azure OpenAI, Anthropic
# direct, ...): leave LLM_GATEWAY_KEY empty and set
# REINVENT4_AGENT_API_KEY instead.
export LLM_GATEWAY_KEY="${LLM_GATEWAY_KEY:-}"
export REINVENT4_AGENT_API_KEY="${REINVENT4_AGENT_API_KEY:-}"
if [ -z "${LLM_GATEWAY_KEY}" ] && [ -z "${REINVENT4_AGENT_API_KEY}" ]; then
    echo "env.sh: set LLM_GATEWAY_KEY (gateway) or REINVENT4_AGENT_API_KEY" >&2
    return 1 2>/dev/null || exit 1
fi

# --- Agent defaults ---------------------------------------------------
# No endpoint is assumed: point REINVENT4_AGENT_BASE_URL at your own LLM
# endpoint, or leave it empty to use the SDK default for the backend.
export REINVENT4_AGENT_BACKEND="${REINVENT4_AGENT_BACKEND:-anthropic}"
export REINVENT4_AGENT_MODEL="${REINVENT4_AGENT_MODEL:-claude-sonnet-4.6}"
export REINVENT4_AGENT_BASE_URL="${REINVENT4_AGENT_BASE_URL:-}"
export REINVENT4_AGENT_NO_VERIFY_SSL="${REINVENT4_AGENT_NO_VERIFY_SSL:-true}"
export REINVENT4_AGENT_THINKING_BUDGET="${REINVENT4_AGENT_THINKING_BUDGET:-1024}"
# --- OpenAI backend (ignored by the anthropic backend) ----------------
# A gateway that publishes OpenAI models as Azure deployments takes the
# gateway root and the deployment id; the backend appends
# /openai/deployments/<model> and api-version=2024-12-01-preview:
#   REINVENT4_AGENT_BACKEND="openai"
#   REINVENT4_AGENT_MODEL="GPT-5.4"             # Azure deployment id
#   REINVENT4_AGENT_BASE_URL="https://llm-gateway.example.com"
# Set API_VERSION only to override that default.
# REASONING_EFFORT (low|medium|high) applies to reasoning models only.
#
# An OpenAI-compatible endpoint that is not an *.openai.azure.com host and
# needs no gateway key (e.g. a private LLM gateway authenticated with
# REINVENT4_AGENT_API_KEY) is called as plain OpenAI — leave API_VERSION
# unset so the base URL is used verbatim:
#   REINVENT4_AGENT_BACKEND="openai"
#   REINVENT4_AGENT_MODEL="gpt-5"
#   REINVENT4_AGENT_BASE_URL="https://llm.example.com/v1"
#   REINVENT4_AGENT_API_KEY="..."
export REINVENT4_AGENT_API_VERSION="${REINVENT4_AGENT_API_VERSION:-}"
export REINVENT4_AGENT_REASONING_EFFORT="${REINVENT4_AGENT_REASONING_EFFORT:-}"

# --- Response token cap -----------------------------------------------
# On reasoning models (gpt-5, o-series) this cap covers the model's hidden
# reasoning tokens as well as the reply, so a cap sized for a chat model
# gets spent entirely on reasoning and turns come back empty — with the
# output token count still rising.  Give those models their full output
# limit; the cap only bounds a response, it is not reserved or billed.
# Other models reject a cap above their own output limit, hence the split.
if [ -z "${REINVENT4_AGENT_MAX_TOKENS:-}" ]; then
    case "${REINVENT4_AGENT_MODEL}" in
        # gpt-5 (and GPT-5.4, gpt-5-mini, ...): 128k output.
        *[Gg][Pp][Tt]-5* | *[Gg][Pp][Tt]5*)
            REINVENT4_AGENT_MAX_TOKENS=128000
            ;;
        # o-series: 100k output.
        [Oo][134] | [Oo][134][-.]*)
            REINVENT4_AGENT_MAX_TOKENS=100000
            ;;
        *)
            REINVENT4_AGENT_MAX_TOKENS=4096
            ;;
    esac
fi
export REINVENT4_AGENT_MAX_TOKENS

# --- Multi-agent (Driver + Reviewer) ----------------------------------
# "true"  -> run both the Driver and an independent read-only Reviewer
#            (enables the /review command and the background Reviewer
#            monitor that critiques running campaigns).
# "false" -> run the Driver agent only.
export REINVENT4_AGENT_MULTI_AGENT="${REINVENT4_AGENT_MULTI_AGENT:-false}"

# --- Logging ----------------------------------------------------------
export REINVENT4_AGENT_LOG_DIR="${REINVENT4_PROJECT_DIR}/logs"

# --- GPU assignments --------------------------------------------------
# REINVENT and Boltz run on separate GPUs.  When Slurm gives you two
# GPUs (e.g. SLURM_STEP_GPUS="0,1"), the first is used for REINVENT
# and the second for Boltz.  Override explicitly if needed.
if [ -n "${SLURM_STEP_GPUS:-}" ]; then
    export REINVENT_GPU="${REINVENT_GPU:-${SLURM_STEP_GPUS%%,*}}"
    export BOLTZ_GPU="${BOLTZ_GPU:-${SLURM_STEP_GPUS#*,}}"
else
    export REINVENT_GPU="${REINVENT_GPU:-0}"
    export BOLTZ_GPU="${BOLTZ_GPU:-1}"
fi

# --- Slurm defaults (override via sbatch flags if needed) -------------
export SLURM_PARTITION="${SLURM_PARTITION:-batch}"
export SLURM_CPUS="${SLURM_CPUS:-4}"
export SLURM_MEM="${SLURM_MEM:-16G}"
export SLURM_TIME="${SLURM_TIME:-4:00:00}"
