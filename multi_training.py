import os, random
from midi_to_statematrix import *
from data import *
import cPickle as pickle
import time

import signal

batch_width = 20 # number of sequences in a batch
batch_len = 16*2 # length of each sequence
division_len = 16 # interval between possible start locations
ofile = "final"

def loadPieces(dirpath):
    pieces = {}

    for fname in os.listdir(dirpath):
        if fname[-4:] not in ('.mid','.MID'):
            continue

        name = fname[:-4]

        outMatrix = midiToNoteStateMatrix(os.path.join(dirpath, fname))
        if len(outMatrix) < batch_len:
            continue

        pieces[name] = outMatrix
        print "Loaded {}".format(name)

    return pieces

def getPieceSegment(pieces):
    piece_output = random.choice(pieces.values())
    start = random.randrange(0,len(piece_output)-batch_len,division_len)
    # print "Range is {} {} {} -> {}".format(0,len(piece_output)-batch_len,division_len, start)

    seg_out = piece_output[start:start+batch_len]
    seg_in = noteStateMatrixToInputForm(seg_out)

    return seg_in, seg_out

def getPieceBatch(pieces):
    i,o = zip(*[getPieceSegment(pieces) for _ in range(batch_width)])
    return numpy.array(i), numpy.array(o)

def trainPiece(model,pieces,epochs,ofile,start=0):
    if not os.path.exists('output/{}'.format(ofile)):
        os.makedirs('output/{}'.format(ofile))
    with open('output/{}/loss.txt'.format(ofile), 'w+') as f:
        f.write('iter,time,loss\n')
    stopflag = [False]
    def signal_handler(signame, sf):
        stopflag[0] = True
    old_handler = signal.signal(signal.SIGINT, signal_handler)
    error = 0
    start_time = time.time()
    for i in range(start,start+epochs):
        if stopflag[0]:
            break
        error = model.update_fun(*getPieceBatch(pieces))
        print "epoch {}, error={}".format(i,error)
        if i % 100 == 0:
            with open('output/{}/loss.txt'.format(ofile), 'a') as f:
                f.write('{},{},{}\n'.format(i, time.time()-start_time, error))
            xIpt, xOpt = map(numpy.array, getPieceSegment(pieces))
            noteStateMatrixToMidi(numpy.concatenate((numpy.expand_dims(xOpt[0], 0), model.predict_fun(batch_len*4, 1, xIpt[0])), axis=0),'output/{}/sample{}'.format(ofile,i))
            if i % 500 == 0:
                pickle.dump(model.learned_config,open('output/{}/params{}_loss{}.p'.format(ofile,i,str(error)), 'wb'))
    signal.signal(signal.SIGINT, old_handler)
    pickle.dump(model.learned_config,open('output/{}/config.p'.format(ofile), 'wb'))
    return error
