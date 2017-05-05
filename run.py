import sys
import model
import multi_training
import main

if len(sys.argv) < 7:
    print("Expected python run.py l1, l2, l3, l4, drp, itr.")
    print("Instead got",sys.argv)
    exit(0)

l1, l2, l3, l4 = [int(x) for x in sys.argv[1:-3]]
drp = float(sys.argv[-3])
itr = int(sys.argv[-2])
ofile = sys.argv[-1]

print("Creating model...")
m = model.Model([l1,l2],[l3,l4],dropout=drp)
print("Model created")
pcs = multi_training.loadPieces("music")
error = multi_training.trainPiece(m,pcs,itr,ofile)
main.gen_adaptive(m,pcs,10,name="{}_{}_{}_{}_{}_{}".format(l1,l2,l3,l4,drp,error))
