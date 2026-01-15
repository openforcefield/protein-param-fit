BEGIN {
    iter = -1
    split("99478323 99478356 99478357 99478358 99478359 99478377 99478378 99478379 99478380 99478418 99478419 99478420 99478421 99478450 99478504 99478505 99478597 99478598 99478599 99478630 99478688 99478689 99478690 99478719 99478752 99478784", backbone_ids)
    split("103670809 103670810 103670811 103670812 103670813 103670814 103670815 103670816 103670817 103670818 103670819 103670820 103670821 103670822 103670823 103670824 103670825 103670826 103670827 103670828 103670829 103670830 103670831 103670832 103670833 103670834 103670835 103670836 103670837 103670838 103670839 103670840 103670841 103670842 103670843 103670844 103670845 103670846 103670847 103670848 103670849 103670850 103670851 103670852 103670853 103670854", sidechain_ids)
    for (i=1; i<=length(backbone_ids); ++i) backbone_td[backbone_ids[i]]
    for (i=1; i<=length(sidechain_ids); ++i) sidechain_td[sidechain_ids[i]]
} /Iteration/ {
    # iter = substr($4, 1, length($4) - 1)
    ++iter
} breakdown && $1 == "Total" {
    objective[iter] = $3
    breakdown = 0
} breakdown && $1 == "Regularization" {
    regularization[iter] = $5
} breakdown && $1 ~ /ab-initio/ {
    split($1, id, "-")
    if (id[3] in backbone_td) torsion_backbone[iter] += $5
    else if (id[3] in sidechain_td) torsion_sidechain[iter] += $5
    else if ($1 ~ /ab-initio-batch/) abinitio[iter] += $5
    else torsion_sm[iter] += $5
} breakdown && $1 ~ /opt-geo/ {
    opt_geo[iter] += $5
} /Objective Function Breakdown/ {
    breakdown = 1
} /Hessian/ {
    ++hessian[iter]
} /successfully/ {
    for (i = 1; i <= NF; ++i) {
        if ($i == "successfully") {
            task = $(i - 4)
            if (task ~ /torsion/) ++torsion[iter]
            else if (task ~ /opt-geo/) ++opt[iter]
        }
    }
} END {
    for (i=0; i<=iter; ++i) {
        out = "Iter %2d Obj %8.2f BB %7.2f SC %7.2f 4M %7.2f SM %7.2f "
        out = out "Opt-Geo %7.2f Reg %7.2f Hess %2d TD %4d Opt-Geo %3d\n"
        printf out, i, objective[i], torsion_backbone[i], \
            torsion_sidechain[i], abinitio[i], torsion_sm[i], opt_geo[i], \
            regularization[i], hessian[i], torsion[i], opt[i]
    }
}

