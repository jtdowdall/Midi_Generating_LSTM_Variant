<span class="ptmr7t-x-x-240">Gate-Variant of Long Short Term Memory
(LSTM)</span> <span class="ptmr7t-x-x-240">Neural Networks for Music
Generation</span>

<span class="ptmr7t-x-x-110">Jonathan Dowdall and Fathi M. Salem</span>
College of Engineering Michigan State University East Lansing, Michigan
48825 Email: <span class="ptmri7t-">dowdallj@msu.edu
</span>$\parallel \textit{salemf@msu.edu}$

<span class="ptmb7t-x-x-90">Abstract</span> <span
class="ptmr7t-x-x-90">This paper evaluates a variant of Long Short Term
Memory (LSTM) </span><span class="ptmr7t-x-x-90"> cells by reducing the
number of parameters in the</span> <span class="ptmr7t-x-x-90">input,
forget, </span><span class="ptmr7t-x-x-90"> and update gates. The LSTM
variant demonstrates its ability to compose novel polyphonic music
similar to the</span> <span class="ptmr7t-x-x-90">original LSTM model
with significantly fewer parameters.</span> <span id="x1-2r1"></span>

<span class="ptmrc7t-">I. I<span class="small-caps">n</span><span
class="small-caps">t</span><span class="small-caps">r</span><span
class="small-caps">o</span><span class="small-caps">d</span><span
class="small-caps">u</span><span class="small-caps">c</span><span
class="small-caps">t</span><span class="small-caps">i</span><span
class="small-caps">o</span><span class="small-caps">n</span></span>
<span id="Q1-1-0"></span>

Recurrent neural networks (RNN) have fundamentally changed the  role of
artificial neural networks. Through recurrent connections,  neural
networks are capable of learning with temporal understanding.  The gap
between artificial and human intelligence continues to shrink as
 recurrent neural networks conquer tasks from language modeling <span
class="cite"> \[<span class="ptmb7t-">?</span>\]</span> to  stock price
pattern recognition <span class="cite"> \[<span
class="ptmb7t-">?</span>\]</span>. Furthermore, the  implementation of
memory gates in the LSTM as described by Hochreiter et al <span
class="cite">  \[<span class="ptmb7t-">?</span>\]</span> made it
practical for RNNs to  capture longer term dependencies, effectively
maintaining memory not too unlike human intelligence. LSTM networks have
demonstrated novel applications such as drawing <span
class="cite"> \[<span class="ptmb7t-">?</span>\]</span> and composing
music <span class="cite">  \[<span class="ptmb7t-">?</span>\]</span>.

LSTM cells use gating network signals to control the retention of memory
 and the influence of new input. The three gates- input, forget, and
 output- each contain as many parameters as the memory cell they
regulate.  Dey and Salem <span class="cite"> \[<span
class="ptmb7t-">?</span>\]</span> have shown that gated recurrent neural
networks can acheive similar performance at a fraction of the
computational expense by reducing the number of parameters in the memory
gates. This was demonstrated on Gated Recurrent Units, a gated RNN
similar to the LSTM except with only two memory gates. The proposed GRU
variants were able to match and even outperform the original GRU on the
MNIST dataset of hand-written digits.

This paper aims to demonstrate that LSTM networks can also benefit from
the parameter reduction method shown in <span class="cite"> \[<span
class="ptmb7t-">?</span>\]</span>. The LSTM and its variant are
implemented in otherwise identical Biaxial LSTM (BALSTM) networks.
Introduced by Johnson <span class="cite"> \[<span
class="ptmb7t-">?</span>\]</span>, the BALSTM has demonstrated a very
impressive ability to recognize patterns in timing and melodic structure
from midi compositions. Replacing the LSTM with the LSTM variant results
in similar convergence with a shorter training time.

<span id="x1-3r2"></span>

<span class="ptmrc7t-">II. B<span class="small-caps">a</span><span
class="small-caps">c</span><span class="small-caps">k</span><span
class="small-caps">g</span><span class="small-caps">r</span><span
class="small-caps">o</span><span class="small-caps">u</span><span
class="small-caps">n</span><span class="small-caps">d</span></span>
<span id="Q1-1-0"></span>

<span id="x1-4r1"></span>

<span class="ptmri7t-">A. LSTM</span> <span id="Q1-1-0"></span>

The simple RNN model computes hidden activation as

  --------------------------------------------------------------------------------- -----
  <span id="x1-5r1"></span> $$h_{t} = g\left( {Wx_{t} + Uh_{t - 1} + b} \right)$$   (1)
  --------------------------------------------------------------------------------- -----

where $x_{t}$ is the <span class="ptmri7t-">m-dimensional</span> input
vector at time $t$, $h_{t}$ is the <span
class="ptmri7t-">n-dimensional</span> hidden state, $g$ is the
activation function commonly the logistic (sigmoid) or hyperbolic
tangent function. $W$, $U$, and $b$ are the parameters which are
respectively sized $n \times m$, $n \times n$, and $n \times 1$.

In order to accommodate for exploding and vanishing gradients, the LSTM
RNN architecture introduces three memory gates: input $i$, forget $f$,
and ouput $o$, which regulate the final output (state) of the memory
cell as follows:

  ------------------------------------------------------------------------------------------------- -----
  <span id="x1-6r2"></span> $$\tilde{c} = g\left( {W_{c}x_{t} + U_{c}h_{t - 1} + b_{c}} \right)$$   (2)
  ------------------------------------------------------------------------------------------------- -----

  ------------------------------------------------------------------------------------- -----
  <span id="x1-7r3"></span> $$c_{t} = f_{t} \odot c_{t - 1} + i_{t} \odot \tilde{c}$$   (3)
  ------------------------------------------------------------------------------------- -----

  --------------------------------------------------------- -----
  <span id="x1-8r4"></span> $$h_{t} = o_{t} \odot c_{t}$$   (4)
  --------------------------------------------------------- -----

The intermediate candidate c̃ for the cell reflects Eqn (1). In Eqn (2),
we compute the internal hidden state $c$ as a weighted average of
$\tilde{c}$ and the previous internal memory state $c_{t - 1}$. The
forget gate $f_{t}$ and the input gate $i_{t}$ give weight to the old
and new hidden states respectively, via element-wise multiplication.
Finally, the output gate $o$ is multiplied by the internal hidden state
of the cell to produce a final activation output $h_{t}$.

  -------------------------------------------------------------------------------------------------- -----
  <span id="x1-9r5"></span> $$i_{t} = \sigma\left( {W_{i}x_{t} + U_{i}h_{t - 1} + b_{i}} \right)$$   (5)
  -------------------------------------------------------------------------------------------------- -----

  --------------------------------------------------------------------------------------------------- -----
  <span id="x1-10r6"></span> $$f_{t} = \sigma\left( {W_{f}x_{t} + U_{f}h_{t - 1} + b_{f}} \right)$$   (6)
  --------------------------------------------------------------------------------------------------- -----

  --------------------------------------------------------------------------------------------------- -----
  <span id="x1-11r7"></span> $$o_{t} = \sigma\left( {W_{o}x_{t} + U_{o}h_{t - 1} + b_{o}} \right)$$   (7)
  --------------------------------------------------------------------------------------------------- -----

Each gate contains its own set of parameters $W$ and $U$ which are
adaptively updated at each training step. This means the LSTM model has
4 times the amount of parameters as the simple RNN model shown in Eqn
(1). This is a considerable increase in computation.

<span id="x1-12r2"></span>

<span class="ptmri7t-">B. Gate Variant</span> <span id="Q1-1-0"></span>

All of the gates’ parameters are updated with the same information
pertaining to the state of the overall network. This leads to a
redundancy in the signals driving the gating signals <span
class="cite"> \[<span class="ptmb7t-">?</span>\]</span>. The proposed
variant modifies the LSTM so that each gate is computed using only the
previous hidden state.

  --------------------------------------------------------------------------------------------------- -----
  <span id="x1-13r8"></span> $$i_{t} = \sigma\left( {W_{i}x_{t} + U_{i}h_{t - 1} + b_{i}} \right)$$   (8)
  --------------------------------------------------------------------------------------------------- -----

  --------------------------------------------------------------------------------------------------- -----
  <span id="x1-14r9"></span> $$f_{t} = \sigma\left( {W_{f}x_{t} + U_{f}h_{t - 1} + b_{f}} \right)$$   (9)
  --------------------------------------------------------------------------------------------------- -----

  ---------------------------------------------------------------------------------------------------- ------
  <span id="x1-15r10"></span> $$o_{t} = \sigma\left( {W_{o}x_{t} + U_{o}h_{t - 1} + b_{o}} \right)$$   (10)
  ---------------------------------------------------------------------------------------------------- ------

This modified architecture should be much less computationally expensive
without significant decrease in performance.

<span id="x1-16r3"></span>

<span class="ptmri7t-">C. Experiment</span> <span id="Q1-1-0"></span>

In order to generate music, the network takes midi input, a binary
representation of notes and timings. There are 88 keys on a piano, so
the network contains 4 layers of 88 stacked LSTMs, forming a BALSTM. The
details of this implementation can be found in <span
class="cite"> \[<span class="ptmb7t-">?</span>\]</span>. The final layer
is activated by the logistic function to indicate whether or not to play
a note or hold a note if it is already being articulated. The goal is to
produce interesting music, so there is no accuracy measure. However,
log-likelihood is used to calculate loss at each step for adaptive
parameter updates.

------------------------------------------------------------------------

<div class="figure">

![PIC](balstm.png) <span id="x1-17r1"></span> <span
class="ptmr7t-x-x-80">Fig.</span><span
class="ptmr7t-x-x-80"> 1.</span><span
class="ptmr7t-x-x-80"> </span><span class="ptmr7t-x-x-80"> Network
architecture</span>

</div>

------------------------------------------------------------------------

Two models were trained: one with normal LSTM cells, and one with the
LSTM variant.

------------------------------------------------------------------------

<div class="float">

</div>
