#!/usr/bin/env bash

# result placed in kernels
_easypap_kernels()
{
    local f tmp

    kernels=

    if [ ! -f obj/none.o ]; then
        return
    fi

    tmp=`nm obj/*.o | awk '/_compute_seq$/ {print $3}'`

    for f in $tmp; do
        f=${f#_}
        f=${f%_compute_seq}
        kernels="$kernels $f"
    done
}

# result placed in variants
_easypap_variants()
{
    local f tmp

    variants=

    if (( $# == 0 )); then
        set none
    fi

    # Check if easypap is compiled
    if [ ! -f obj/$1.o ]; then
        return
    fi

    tmp=`nm obj/$1.o | awk '/ +_?'"$1"'_compute_[^.]*$/ {print $3}'`

    for f in $tmp; do
        variants="$variants ${f#*_compute_}"
    done
}

# result placed in ovariants
_easypap_ocl_variants()
{
    local f tmp k

    ovariants=

    if (( $# == 0 )); then
        set none
    fi

    if [[ ! -f kernel/ocl/$1.cl ]]; then
        return
    fi

    # The most secure way of finding OpenCL kernels is to compile the .cl file
    tmp=`./run -k $1 -lov`

    # But the "grep into the .cl file" method is much much faster!
    #tmp=`awk '/__kernel/ {print $3}' < kernel/ocl/${k}.cl`

    for f in $tmp; do
        if [[ $f =~ .*update_texture$ || $f =~ bench_kernel ]]; then
            continue
        fi
        ovariants="$ovariants ${f#$1_}"
    done
}

# result placed in draw_funcs
_easypap_draw_funcs()
{
    local f tmp

    draw_funcs=

    if (( $# == 0 )); then
        set none
    fi

    # Check if easypap is compiled
    if [ ! -f obj/$1.o ]; then
        return
    fi

    tmp=`nm obj/$1.o | awk '/ +_?'"$1"'_draw_[^.]*$/ {print $3}'`

    for f in $tmp; do
        draw_funcs="$draw_funcs ${f#*_draw_}"
    done
}

# result placed in pos
_easypap_option_position()
{
    local p

    if [[ $1 =~ ^--.* ]]; then
	    list=("${LONG_OPTIONS[@]}")
    else
	    list=("${SHORT_OPTIONS[@]}")
    fi
    for (( p=0; p < $NB_OPTIONS; p++ )); do
	    if [[ "${list[p]}" = "$1" ]]; then
	        pos=$p
	        return
	    fi
    done
    pos=$NB_OPTIONS
}

_easypap_remove_from_suggested()
{
    local i
    for (( i=0; i < ${#suggested[@]}; i++ )); do 
        if [[ ${suggested[i]} == $1 ]]; then
            suggested=( "${suggested[@]:0:$i}" "${suggested[@]:$((i + 1))}" )
            i=$((i - 1))
        fi
    done
}

_easypap_option_suggest()
{
    local c e o suggested=("$@")

    # We should prune the options than should only appear at first position    
    for (( o=0; o < $NB_OPTIONS; o++ )); do
        local short=${SHORT_OPTIONS[o]}
        local only_in_first_place_opt=only_in_first_place_${short#-}
        eval only_in_first_place_opt="\$${only_in_first_place_opt}"
        if (( only_in_first_place_opt == 1 )); then
            _easypap_remove_from_suggested ${LONG_OPTIONS[o]}
            _easypap_remove_from_suggested $short
           fi
    done

    for (( c=1; c < $COMP_CWORD; c++ )); do
	    if [[ ${COMP_WORDS[c]} =~ ^-.* ]]; then
	        _easypap_option_position ${COMP_WORDS[c]}
	        if (( pos < NB_OPTIONS )); then
	            local short=${SHORT_OPTIONS[pos]}
                local multiple_opt=multiple_${short#-}
                eval multiple_opt="\$${multiple_opt}"

                if [[ -z $multiple_opt ]]; then
		            # we shall remove this option from suggested options
                    _easypap_remove_from_suggested ${LONG_OPTIONS[pos]}
                    _easypap_remove_from_suggested $short
                fi

		        # also remove antagonist options
		        local excluding_opt=exclude_${short#-}
		        eval excluding_opt='(${'${excluding_opt}'[@]})'
		        for ((e=0; e < ${#excluding_opt[@]}; e++ )); do
		            local p=${excluding_opt[e]}
                    _easypap_remove_from_suggested ${LONG_OPTIONS[p]}
                    _easypap_remove_from_suggested ${SHORT_OPTIONS[p]}
		        done
	        fi
	    fi
    done

    COMPREPLY=($(compgen -W '"${suggested[@]}"' -- $cur))
}
