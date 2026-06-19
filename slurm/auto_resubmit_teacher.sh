#!/bin/bash
# Auto-resubmission chain for the single-GPU OGAP teacher (Trillium 24h cap).
#
# Submits DEPTH teacher jobs, each --dependency=afterany the previous one, so the
# teacher continues across 24h windows with no manual resubmits. Every job goes
# through workflow/launch_ogap_train.sh (teacher-only) so its settings are IDENTICAL
# to the first run, and resumes from resume_teacher.pth. The teacher sbatch's
# completion guard makes any trailing job (after epoch>=target) exit instantly,
# so over-provisioning DEPTH is harmless.
#
# Usage:
#   START_AFTER=<running_teacher_jobid> DEPTH=16 bash slurm/auto_resubmit_teacher.sh
#   # START_AFTER empty -> the first job starts immediately (no dependency).
#
# To STOP the whole chain later: scancel every job id printed below.
set -euo pipefail

DEPTH="${DEPTH:-16}"
START_AFTER="${START_AFTER:-}"
LAUNCH="${LAUNCH:-/project/def-rdiaz/ilyaso/OGAP/Scripts/workflow/launch_ogap_train.sh}"
MODE="${MODE:-unet}"

prev="${START_AFTER}"
ids=()
for i in $(seq 1 "${DEPTH}"); do
  dep=""
  [[ -n "${prev}" ]] && dep="afterany:${prev}"
  out="$(RUN_STUDENT=0 RUN_EXPORT=0 RUN_EVAL=0 RUN_OOD=0 RUN_NAS=0 USE_DDP_TEACHER=0 \
         TEACHER_DEPENDENCY="${dep}" bash "${LAUNCH}" "${MODE}" 2>&1)"
  jid="$(printf '%s\n' "${out}" | sed -n 's/^teacher_job=//p' | tail -1)"
  if [[ -z "${jid}" ]]; then
    echo "[chain] FAILED to submit link ${i}; launcher output:" >&2
    printf '%s\n' "${out}" >&2
    exit 1
  fi
  echo "[chain] link ${i}/${DEPTH}: teacher job ${jid}  (dependency=${dep:-none})"
  ids+=("${jid}")
  prev="${jid}"
done

echo "[chain] submitted ${DEPTH} chained teacher jobs."
echo "[chain] job ids: ${ids[*]}"
echo "[chain] to stop the whole chain:  scancel ${ids[*]}"
