#!/usr/bin/env bash

_dir=`dirname $BASH_SOURCE`
_dir=`dirname $_dir`

. ${_dir}/script/easypap-utilities.bash

unset _dir

_easypap_completions()
{
    local LONG_OPTIONS=("--help" "--load-image" "--size" "--kernel" "--variant" "--monitoring" "--thumbnails"
                        "--trace" "--no-display" "--iterations" "--nb-tiles" "--tile-size" "--arg" "--first-touch"
                        "--label" "--mpirun" "--soft-rendering" "--show-ocl" "--tile-width" "--tile-height")
    local SHORT_OPTIONS=("-h" "-l" "-s" "-k" "-v" "-m" "-tn"
                         "-t" "-n" "-i" "-nt" "-ts" "-a" "-ft"
                         "-lb" "-mpi" "-sr" "-so" "-tw" "-th")
    local NB_OPTIONS=${#LONG_OPTIONS[@]}

    local exclude_s=(1) # size excludes load-image
    local exclude_l=(2) # load-image excludes size
    local exclude_n=(5) # no-display excludes monitoring
    local exclude_t=(5) # trace excludes monitoring
    local exclude_m=(7 8) # monitoring excludes trace and no-display
    local exclude_ts=(10 18 19) # tile-size excludes nb-tiles, tile-width and tile-height
    local exclude_nt=(11 18 19) # nb-tiles excludes tile-size, tile-width and tile-height
    local exclude_tw=(10 11) # tile-width excludes nb-tiles and tile-size
    local exclude_th=(10 11) # tile-height excludes nb-tiles and tile-size
    local only_in_first_place_h=1 # --help should only be suggested as the very first argument position
    local only_in_first_place_so=1 # --show-ocl should only be suggested as the very first argument position

    local i cur=${COMP_WORDS[COMP_CWORD]}

    if [[ ${COMP_CWORD} = 1 ]]; then
        if [[ $cur =~ ^--.* ]]; then
            COMPREPLY=($(compgen -W '"${LONG_OPTIONS[@]}"' -- $cur))
        else
            COMPREPLY=($(compgen -W '"${SHORT_OPTIONS[@]}"' -- $cur))
        fi
    else
        prev=${COMP_WORDS[COMP_CWORD-1]}
        case $prev in
            -s|--size)
                COMPREPLY=($(compgen -W "512 1024 2048 4096" -- $cur))
                ;;
            -nt|--nb-tiles)
                COMPREPLY=($(compgen -W "8 16 32 64" -- $cur))
                ;;
            -ts|--tile-size)
                COMPREPLY=($(compgen -W "8 16 32 64" -- $cur))
                ;;
            -tw|--tile-width)
                COMPREPLY=($(compgen -W "4 8 16 32 64" -- $cur))
                ;;
            -th|--tile-height)
                COMPREPLY=($(compgen -W "4 8 16 32 64" -- $cur))
                ;;
            -l|--load-image)
                compopt -o filenames
                if [[ -z "$cur" ]]; then
                    COMPREPLY=($(compgen -f -X '!*.png' -- "images/"))
                else
                    COMPREPLY=($(compgen -o plusdirs -f -X '!*.png' -- $cur))
                fi
                ;;
            -a|--arg)
                local k=
                # search for kernel name
                for (( i=1; i < COMP_CWORD; i++ )); do
                    case ${COMP_WORDS[i]} in
                        -k|--kernel)
                            if (( i < COMP_CWORD - 1)); then
                                k=${COMP_WORDS[i+1]}
                            fi
                            ;;
                        *)
                            ;;
                    esac
                done
                # kernel-specific draw functions (note: will use the 'none' kernel by default)
                _easypap_draw_funcs $k
                if [[ -z "$draw_funcs" ]]; then
                    compopt -o filenames
                    if [[ -z "$cur" ]]; then
                        COMPREPLY=($(compgen -f -- "data/"))
                    else
                        COMPREPLY=($(compgen -f -- "$cur"))
                    fi                
                else
                    COMPREPLY=($(compgen -W "$draw_funcs" -- $cur))
                fi
                ;;
            -k|--kernel)
                _easypap_kernels
                COMPREPLY=($(compgen -W "$kernels" $cur))
                ;;
            -v|--variant)
                local k=
                local ocl=
                # search for kernel name
                for (( i=1; i < COMP_CWORD; i++ )); do
                    case ${COMP_WORDS[i]} in
                        -k|--kernel)
                            if (( i < COMP_CWORD - 1)); then
                                k=${COMP_WORDS[i+1]}
                            fi
                            ;;
                        -o|--ocl)
                            ocl=1
                            ;;
                        *)
                            ;;
                    esac
                done
                if [[ -z $ocl ]]; then
                    # CPU variants (note: will use 'none' if $k is empty)
                    _easypap_variants $k
                    COMPREPLY=($(compgen -W "$variants" -- $cur))
                else
                    # OpenCL variants (note: will use 'none' if $k is empty)
                     _easypap_ocl_variants $k
                     COMPREPLY=($(compgen -W "$ovariants" -- $cur))
                fi
                ;;
            -mpi|--mpirun)
                if [[ -z "$cur" ]]; then
                    COMPREPLY=("\"${MPIRUN_DEFAULT:-"-np 2"}\"")
                fi
                ;;
            -n|--no-display|-m|--monitoring|-t|--trace|-th|--thumbs|-ft|--first-touch|-du|--dump|-p|--pause|-sr|--soft-rendering|-o|--ocl)
                # After options taking no argument, we can suggest another option
                if [[ "$cur" =~ ^--.* ]]; then
                    _easypap_option_suggest "${LONG_OPTIONS[@]}"
                else
                    _easypap_option_suggest "${SHORT_OPTIONS[@]}"
                fi
                ;;
            -*)
                # For remaining options with one argument, we don't suggest anything
                ;;
            *)
                if [[ "$cur" =~ ^--.* ]]; then
                    _easypap_option_suggest "${LONG_OPTIONS[@]}"
                else
                    _easypap_option_suggest "${SHORT_OPTIONS[@]}"
                fi
                ;;
        esac
    fi
}

_easyview_completions()
{
    local LONG_OPTIONS=("--compare" "--no-thumbs" "--help" "--range" "--dir" "--align" "--iteration" "--whole-trace" "--brightness")
    local SHORT_OPTIONS=("-c" "-nt" "-h" "-r" "-d" "-a" "-i" "-w" "-b")
    local NB_OPTIONS=${#LONG_OPTIONS[@]}

    local exclude_r=(6 7) # range excludes iteration and whole-trace
    local exclude_i=(3 7) # iteration excludes range and whole-trace
    local exclude_w=(3 6) # whole-trace excludes range and iteration
    local exclude_nt=(8)  # no-thumbs excludes brightness
    local exclude_b=(1)   # brightness excludes no-threads
    local multiple_d=1    # -d can appear multiple times

    local cur=${COMP_WORDS[COMP_CWORD]}

    if (( COMP_CWORD > 1 )); then
        case ${COMP_WORDS[COMP_CWORD-1]} in
            -d|--dir)
                #compopt -o filenames
                if [[ -z "$cur" ]]; then
                    COMPREPLY=($(compgen -d -- "traces/data/"))
                else
                    COMPREPLY=($(compgen -d -- "$cur"))
                fi
                return
                ;;
            -b|--brightness)
                COMPREPLY=($(compgen -W "128 255" -- $cur))
                return
                ;;
            *)
                ;;        
        esac
    fi

    # check cmdline
    local i tracedir compare=0
    for (( i=1; i < COMP_CWORD; i++ )); do
        case ${COMP_WORDS[i]} in
            -c|--compare)
                compare=1
                ;;
            -d|--dir)
                if (( i < COMP_CWORD - 1)); then
                    tracedir=${COMP_WORDS[i+1]}
                fi
                ;;
            *)
                ;;
        esac
    done
    if [[ -z $tracedir ]]; then
        tracedir="traces/data"
    fi
    if [[ "$cur" =~ ^--.* ]]; then
        _easypap_option_suggest "${LONG_OPTIONS[@]}"
    elif [[ "$cur" =~ ^-.* || $compare = 1 ]]; then
        _easypap_option_suggest "${SHORT_OPTIONS[@]}"
    else
        compopt -o filenames
        if [[ -z "$cur" ]]; then
            COMPREPLY=($(compgen -f -X '!*.evt' -- ${tracedir}/))
        else
            COMPREPLY=($(compgen -o plusdirs -f -X '!*.evt' -- $cur))
        fi
    fi
}

complete -F _easypap_completions ./run
complete -F _easyview_completions ./view
