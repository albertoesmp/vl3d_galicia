D=(1 2 3 4 5 6 7 8 9 10)
L=(1 2 3 4 5 6 7 8 9 10)
n=${#D[@]}
n_minus_one=$(( ${n} - 1 ))

for i in $(seq 0 ${n_minus_one}); do
    for j in $(seq 0 ${n_minus_one}); do
        cat << EOF
    ),(
        'KPConv_d${D[$i]}_${L[$j]} init',
        'Initialized KPConv layer ${L[$j]} at depth ${D[$i]}.'
    ),(
        'SKPConv_d${D[$i]}_${L[$j]} init',
        'Initialized strided KPConv layer ${L[$j]} at depth ${D[$i]}.'
    ),(
        'KPConv_d${D[$i]}_${L[$j]} trained',
        'Trained KPConv layer ${L[$j]} at depth ${D[$i]}.'
    ),(
        'SKPConv_d${D[$i]}_${L[$j]} trained',
        'Trained strided KPConv layer ${L[$j]} at depth ${D[$i]}.'
    ),(
        'KPConv_d${D[$i]}_${L[$j]} diff',
        'Training update (difference) of the KPConv layer ${L[$j]} at depth ${D[$i]}.'
    ),(
        'SKPConv_d${D[$i]}_${L[$j]} diff',
        'Training update (difference) of the strided KPConv layer ${L[$j]} at depth ${D[$i]}.'
    ),(
        'KPConv_d${D[$i]}_${L[$j]} init hist',
        'Histogram of initialized KPConv layer ${L[$j]} at depth ${D[$i]}.'
    ),(
        'SKPConv_d${D[$i]}_${L[$j]} init hist',
        'Histogram of initialized strided KPConv layer ${L[$j]} at depth ${D[$i]}.'
    ),(
        'KPConv_d${D[$i]}_${L[$j]} trained hist',
        'Histogram of trained KPConv layer ${L[$j]} at depth ${D[$i]}.'
    ),(
        'SKPConv_d${D[$i]}_${L[$j]} trained hist',
        'Histogram of trained strided KPConv layer ${L[$j]} at depth ${D[$i]}.'
    ),(
        'KPConv_d${D[$i]}_${L[$j]} diff hist',
        'Histogram of training update (difference) of the KPConv layer ${L[$j]} at depth ${D[$i]}.'
    ),(
        'SKPConv_d${D[$i]}_${L[$j]} diff hist',
        'Histogram of training update (difference) of the strided KPConv layer ${L[$j]} at depth ${D[$i]}.'
    ),(
        'KPConv_d${D[$i]}_${L[$j]} init Q',
        'Structure space of the initialized KPConv layer ${L[$j]} at depth ${D[$i]}.'
    ),(
        'SKPConv_d${D[$i]}_${L[$j]} init Q',
        'Structure space of the initialized strided KPConv layer ${L[$j]} at depth ${D[$i]}.'
    ),(
        'KPConv_d${D[$i]}_${L[$j]} trained Q',
        'Structure space of the trained KPConv layer ${L[$j]} at depth ${D[$i]}.'
    ),(
        'SKPConv_d${D[$i]}_${L[$j]} trained Q',
        'Structure space of the trained strided KPConv layer ${L[$j]} at depth ${D[$i]}.'
EOF
    done
done
