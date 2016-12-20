## What is this?

Currently an implementation of [multiple viewpoint
systems](http://www.tandfonline.com/doi/abs/10.1080/09298219508570672) for music
generation.

## What is a multiple viewpoint system?

__tl;dr__: in general, an ensemble of higher-order Markov chains, each modelling a specific
attribute of a sequence. 

A popular and simple way of generating or predicting music involves using
[Markov chains](https://en.wikipedia.org/wiki/Markov_chain). Applied to melody
generation, a Markov chain calculates the probability of each possible note
occurring after the current one. This probability distribution is based _solely_
on the current event. By iteratively sampling from this distribution, one can
generate a melody.

Context models are a higher-order generalisation of Markov chains: they look at
more than one element of history. For example, an order-2 context model over
notes would predict a different distribution for the context `{G,A}` from that
which it would predict for `{G,B}`. __TODO__: example here.

In the context of music, a _viewpoint_ can be thought of as a certain musical
attribute that varies over time which we might want to model. For example, pitch
and duration would be basic examples of viewpoints. One problem with modelling
pitch alone is that pitch does not capture the relationship between notes which
is invariant between keys. An _interval_ viewpoint, looking at the melodic
interval between consecutive notes _does_ capture this relationship, however.
Such a viewpoint, since it is derived from pitch, can also be used to predict
pitch.

By associating each viewpoint with a context model and combining thier
predictions, one obtains a _multiple viewpoint system_. It turns out that
multiple viewpoint systems are objectively better predictors over single
viewpoint systems (i.e. Markov chains): that is, they can predict music with
lower entropy.

## Compilation and Tests

To compile the project you will need the [scons](http://www.scons.org/) build
system. On macOS with [homebrew](http://brew.sh/):
```
brew update
brew install scons
```

or on Ubuntu/Debian:
```
sudo apt update
sudo apt install scons
```

The unit tests are written using [Catch](https://github.com/philsquared/Catch)
and can be run with `scons test`:
```
$ scons -Q test
test/ctx_test.out
===============================================================================
All tests passed (192 assertions in 13 test cases)
```

## LaTeX Output

Objects such as
[viewpoints](https://github.com/alexcoplan/p2proj/blob/master/src/viewpoint.hpp)
and [sequence
models](https://github.com/alexcoplan/p2proj/blob/master/src/sequence_model.hpp#L159)
can dump their internal data structure to LaTeX. In order to compile this LaTeX,
there are a few prerequisites:
 - [dot2tex](https://dot2tex.readthedocs.io/en/latest/): `pip install dot2tex`
   for graph generation.
 - A XeLaTeX compiler to get support for certain fonts for music notation used
   by the [lilyglyphs](https://www.ctan.org/pkg/lilyglyphs?lang=en) package. In
   your `~/.latexmkrc`:
```
$pdflatex = 'xelatex --shell-escape %O %S'; # XeLaTeX for using fontspec/lilglyphs
```
 - The packages `lilyglyphs` for music notation and `dot2texi` for invoking
   `dot2tex` within LaTeX. 

## Corpus Generation

The corpus is compiled using [music21](http://web.mit.edu/music21/). As of this
writing, you will need to install this from source, since the script relies on a
[patch](https://github.com/cuthbertLab/music21/pull/200) I recently submitted to
the project. I recommend using
[virtualenv](https://virtualenv.pypa.io/en/stable/). To generate the chorale
melody corpus,
run
``` 
python script/prepare_chorales.py
```
or just use the [pre-built
corpus](https://github.com/alexcoplan/p2proj/tree/master/src/corpus).
