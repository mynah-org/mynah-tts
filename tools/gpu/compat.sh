# Sourced by every tools/gpu script: accept the old MYNAH_L4_* names (the
# harness was first written for an L4) as aliases of MYNAH_GPU_*.
for __v in $(compgen -e | grep '^MYNAH_L4_'); do
  __n="MYNAH_GPU_${__v#MYNAH_L4_}"
  [ -z "${!__n+x}" ] && export "$__n=${!__v}"
done
unset __v __n
