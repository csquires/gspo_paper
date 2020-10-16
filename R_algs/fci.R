#!/usr/bin/env Rscript
suppressMessages(library(pcalg))
suppressMessages(library(RcppCNPy))

args = commandArgs(trailingOnly=TRUE)
samples_filename = args[1]
alpha = as.numeric(args[2])
use_fci_plus = args[3] == 'True'
samples = npyLoad(samples_filename)

suffStat = list(C = cor(samples), n = nrow(samples))

PC = FALSE
# run PC for testing purposes
if (PC) {
    res = pc(suffStat, gaussCItest, alpha, p = ncol(samples), verbose=FALSE)
    mag = pag2magAM(res)
    cat(as(mag, 'amat'))
} else {
    if (use_fci_plus){
        res = fciPlus(suffStat, gaussCItest, alpha, p = ncol(samples), verbose=FALSE)
        mag = pag2magAM(res@amat, 1)
        cat(mag)
    } else {
        res = fci(suffStat, gaussCItest, alpha, p = ncol(samples), verbose=FALSE)
        mag = pag2magAM(res@amat, 1, max.chordal=ncol(samples))
        cat(mag)
    }
}

