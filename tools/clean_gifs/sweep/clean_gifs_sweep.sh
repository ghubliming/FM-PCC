#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 2026-09-19 — selective GIF cleanup for logs/, to make room for the Gen15 U17 wave
#
#   CHECK  (default, deletes NOTHING):
#     bash tools/clean_gifs/sweep/clean_gifs_sweep.sh
#     bash tools/clean_gifs/sweep/clean_gifs_sweep.sh --phase 1
#
#   REAL DELETE (only after reading the check output):
#     bash tools/clean_gifs/sweep/clean_gifs_sweep.sh --phase 1 --apply
#     bash tools/clean_gifs/sweep/clean_gifs_sweep.sh --phase 2 --apply
#
# WHY. `logs/` is 97.6 GiB and `.gif` is 35.3 GiB of it (19,210 files, 36% of the tree) —
# the single largest extension, larger than every checkpoint combined. The 19-09 U17 wave
# died of `OSError: [Errno 28] No space left on device` with 2.9 GiB free; eleven of its
# twelve children never started because Slurm could not even create their log files.
# GIFs are rollout RENDERS: no metric reads them, `eval_mix_uav.py` times only the policy
# call, and they regenerate with `record=gif`. They are the cheapest 30 GiB in the tree.
#
# ── THE POLICY (chosen by the author 2026-09-19) ─────────────────────────────
#   PHASE 1  gifs inside (Bf_*) / (Archive*) / (smoke*) / (legacy*) backup dirs
#            -> KEEP 1 per directory. Those folders are dead by convention and are
#               referenced nowhere in DA_in_Paper/ or Writing/ (grepped, 2026-09-19).
#   PHASE 2  gifs in LIVE directories
#            -> KEEP 10% per directory, evenly spaced, never fewer than 1.
#               A percentage, not a fixed count, so a 1080-rollout cell keeps 108 examples
#               and a 3-rollout cell keeps 1.
#
# 🔴 NEVER TOUCHED, and the script refuses to consider them:
#   · uav_expert_data*        the expert DEMONSTRATION renders (~4 GiB). Author's call:
#                             any of them may still become a thesis figure.
#   · */expert_references/*   the one gif class DA_in_Paper sources a figure from
#                             (plotting/sources.py:300,311 — fig_aligning_camera_*).
#   · anything outside logs/  the script resolves its root and refuses to run elsewhere.
#
# Selection is DETERMINISTIC (version-sorted, evenly spaced), so a check run and the
# --apply run that follows it choose exactly the same files.
#
# Every --apply writes a manifest of what it deleted to
#   logs/_clean_gifs_runlogs/<timestamp>_phase<N>.manifest
# so a later "where did that gif go" question has an answer.
#
# Tracked in git (tools/clean_gifs/sweep/) — arrives with a normal `git pull`.
# Companion to tools/clean_gifs/clean_gifs.py (single-run, fixed keep-count); see the README.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

APPLY=0
PHASE=all
while [ $# -gt 0 ]; do
    case "$1" in
        --apply)  APPLY=1 ;;
        --phase)  PHASE="${2:-}"; shift ;;
        --phase=*) PHASE="${1#*=}" ;;
        -h|--help) sed -n '2,45p' "$0"; exit 0 ;;
        *) echo "[FAIL] unknown argument '$1' (use --phase 1|2|all, --apply)"; exit 1 ;;
    esac
    shift
done
case "$PHASE" in 1|2|all) ;; *) echo "[FAIL] --phase must be 1, 2 or all (got '$PHASE')"; exit 1 ;; esac

find_repo_root() {
    local d
    for d in "$PWD" "$(cd "$(dirname "$0")" && pwd)"; do
        while [ "$d" != "/" ] && [ -n "$d" ]; do
            if [ -f "$d/Slurm_Codes/submit.sh" ]; then echo "$d"; return 0; fi
            d="$(dirname "$d")"
        done
    done
    return 1
}
REPO="$(find_repo_root)" || { echo "[FAIL] no Slurm_Codes/submit.sh above \$PWD or this script."; exit 1; }
cd "$REPO"
ROOT="$REPO/logs"
[ -d "$ROOT" ] || { echo "[FAIL] $ROOT does not exist — nothing to clean."; exit 1; }
case "$ROOT" in */FM-PCC/logs) ;; *) echo "[FAIL] refusing to operate on '$ROOT' (not a FM-PCC logs/ dir)"; exit 1 ;; esac
echo "[ ok ] repo root : $REPO"
echo "[ ok ] gif root  : $ROOT"
echo "[ disk ] $(df -h "$ROOT" | awk 'NR==2 {print $4" free on "$6}')"

# ── classification ───────────────────────────────────────────────────────────
# a path is PHASE 1 if any path component marks it a backup/archive snapshot
is_backup() { case "$1" in *"(Bf_"*|*"(Archive"*|*"(smoke"*|*"(legacy"*|*"(Outdated"*|*"(Abandoned"*) return 0 ;; *) return 1 ;; esac; }
# 🔴 hard exclusions — never a candidate, in any phase
is_excluded() { case "$1" in *uav_expert_data*|*/expert_references/*) return 0 ;; *) return 1 ;; esac; }

human() { awk -v b="$1" 'BEGIN{ s="B KiB MiB GiB TiB"; split(s,u," "); i=1;
         while (b>=1024 && i<5){ b/=1024; i++ } printf "%.2f %s", b, u[i] }'; }

TS="$(date +%Y%m%d_%H%M%S)"
# marker dropped into every directory an --apply thinned; see the idempotency guard below
STAMP=".gifs_thinned"
MANIFEST_DIR="$ROOT/_clean_gifs_runlogs"
[ "$APPLY" -eq 1 ] && mkdir -p "$MANIFEST_DIR"

# ── the worker: one phase ────────────────────────────────────────────────────
run_phase() {
    local phase="$1" label keep_desc
    if [ "$phase" = 1 ]; then label="PHASE 1 · gifs inside (Bf_*)/archive dirs"; keep_desc="keep 1 per directory"
    else                       label="PHASE 2 · gifs in LIVE dirs";              keep_desc="keep 10% per directory (evenly spaced, min 1)"; fi

    echo
    echo "=============================================================================="
    echo "$label"
    echo "  policy: $keep_desc"
    [ "$APPLY" -eq 1 ] && echo "  mode  : 🔴 APPLY — files WILL be deleted" || echo "  mode  : CHECK — nothing is deleted"
    echo "=============================================================================="

    local del_list dir_list; del_list="$(mktemp)"; dir_list="$(mktemp)"
    local n_dirs=0 n_total=0 n_keep=0 n_del=0 b_total=0 b_del=0 n_skip=0 n_done=0

    # every directory that directly contains at least one .gif
    while IFS= read -r -d '' d; do
        is_excluded "$d/" && { n_skip=$((n_skip+1)); continue; }
        # 🔴 IDEMPOTENCY GUARD. Without it a second --apply would thin the SURVIVORS again
        # (phase 2 keeps 10%, so a re-run leaves 10% of 10% = 1%). An --apply stamps every
        # directory it touched; a stamped directory is never reconsidered.
        [ -e "$d/$STAMP" ] && { n_done=$((n_done+1)); continue; }
        if [ "$phase" = 1 ]; then is_backup "$d" || continue
        else                      is_backup "$d" && continue; fi

        # version-sorted so rollout_2 < rollout_10 (plain sort would not)
        local gifs=() g
        while IFS= read -r -d '' g; do gifs+=("$g"); done \
            < <(find "$d" -maxdepth 1 -type f -name '*.gif' -print0 | sort -zV)
        local n=${#gifs[@]}
        [ "$n" -eq 0 ] && continue

        local k
        if [ "$phase" = 1 ]; then k=1
        else k=$(( (n + 9) / 10 )); fi      # ceil(n/10), so n>=1 always keeps >=1
        [ "$k" -gt "$n" ] && k=$n

        # evenly spaced keep indices: round(j*n/k) for j = 0 .. k-1
        local keep_flag=() i j idx
        for ((i=0; i<n; i++)); do keep_flag[i]=0; done
        for ((j=0; j<k; j++)); do idx=$(( j * n / k )); keep_flag[idx]=1; done

        n_dirs=$((n_dirs+1))
        printf '%s\0' "$d" >> "$dir_list"
        for ((i=0; i<n; i++)); do
            local sz; sz=$(stat -c %s "${gifs[i]}")
            n_total=$((n_total+1)); b_total=$((b_total+sz))
            if [ "${keep_flag[i]}" -eq 1 ]; then
                n_keep=$((n_keep+1))
            else
                n_del=$((n_del+1)); b_del=$((b_del+sz))
                printf '%s\0' "${gifs[i]}" >> "$del_list"
            fi
        done
    done < <(find "$ROOT" -type f -name '*.gif' -printf '%h\0' | sort -zu)

    echo
    printf '  directories in scope : %d\n' "$n_dirs"
    printf '  gifs found           : %d   (%s)\n' "$n_total" "$(human "$b_total")"
    printf '  KEEP as examples     : %d\n' "$n_keep"
    printf '  DELETE               : %d   (%s)   <-- reclaimed\n' "$n_del" "$(human "$b_del")"
    [ "$n_skip" -gt 0 ] && printf '  excluded dirs        : %d  (uav_expert_data* / expert_references/ — never touched)\n' "$n_skip"
    [ "$n_done" -gt 0 ] && printf '  already thinned      : %d  (carry %s from an earlier --apply; skipped)\n' "$n_done" "$STAMP"

    if [ "$n_del" -eq 0 ]; then
        echo "  nothing to do for this phase."; rm -f "$del_list" "$dir_list"; return 0
    fi

    if [ "$APPLY" -eq 0 ]; then
        echo
        echo "  sample of what WOULD be deleted (first 5):"
        # awk, NOT `head -5`: head closes the pipe early, tr dies of SIGPIPE, and with
        # `set -o pipefail` + `set -e` that aborted the whole script after phase 1.
        tr '\0' '\n' < "$del_list" | awk 'NR<=5 {print "    " $0}'
        echo
        echo "  CHECK ONLY — nothing was deleted."
        echo "  To delete these: bash $0 --phase $phase --apply"
        rm -f "$del_list" "$dir_list"; return 0
    fi

    local man="$MANIFEST_DIR/${TS}_phase${phase}.manifest"
    tr '\0' '\n' < "$del_list" > "$man"
    echo "  manifest: $man"
    xargs -0 rm -f -- < "$del_list"
    while IFS= read -r -d '' d; do
        printf 'thinned %s by clean_gifs_20260919.sh phase %s\n' "$TS" "$phase" > "$d/$STAMP"
    done < "$dir_list"
    rm -f "$del_list" "$dir_list"
    echo "  ✅ deleted $n_del files, reclaimed $(human "$b_del")"
    echo "  stamped $n_dirs directories with $STAMP — a re-run will skip them."
}

if [ "$PHASE" = 1 ] || [ "$PHASE" = all ]; then run_phase 1; fi
if [ "$PHASE" = 2 ] || [ "$PHASE" = all ]; then run_phase 2; fi

echo
echo "──────────────────────────────────────────────────────────────"
echo "[ disk ] $(df -h "$ROOT" | awk 'NR==2 {print $4" free on "$6}')"
if [ "$APPLY" -eq 0 ]; then
    echo "CHECK ONLY — no file was removed."
else
    echo "Empty directories are left in place (they cost nothing and keep the tree shape)."
fi
echo "Reminder — also delete the crashed U17 cell before resubmitting the wave:"
echo "  rm -rf logs/UAV_MIX/uav-pillars/plans/mix_uav_af/*/Eaf_K1_*_u7xl"
