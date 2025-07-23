export CONFIG=experiments/only_segment/ft.cfg
for glottocode in ainu1240 ruul1235 lezg1247 natu1246 nyan1302 dido1241 uspa1245 arap1274
do
    wandb sweep --name "sweep-${glottocode}" --project polygloss-sweeps --entity wav2gloss /Users/michaelginn/Documents/Research/PolyGloss/polygloss/experiments/only_segment/sweep.yaml
done
