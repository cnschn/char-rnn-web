# char-rnn-web
A web service to train, and sample from, character based RNN models.


## Worker container proof-of-concept

For now we only have a *worker* container based on
[ekzhang/char-rnn-keras](https://github.com/ekzhang/char-rnn-keras). To
use it, you need to mount a *session* directory to `/session` which,
as a bare minimum, includes a directory named `data` with a file
`data.txt` inside.

Steps to train on the provided example data:

```
$ cd worker/
$ docker build -t crrn-worker .
[...] # docker build output
$ cd ..
$ docker run --rm -v $(pwd)/session-alice:/session crrn-worker
[...] # tensorflow training output
```

This creates additional directories in the session, `logs` containing
statistics for each trained epoch, so after a while

```
$ cat alice-session/logs/training_log.csv
epoch,loss,acc
1,3.220632553100586,0.16825272142887115
2,2.6479268074035645,0.280619740486145
3,2.153308391571045,0.3988807499408722
4,1.914513111114502,0.458984375
...

```

and stored snapshots in `model` which you can sample from by running

```
$ docker run --rm -v $(pwd)/alice-session:/session crrn-worker python /src/sample.py 38
just as the Dormouse do, sappose; but it was thoughts to be behed
for still and perhaps it over were raboing about over for the mistance, took the chimneys and was swell as she could be quite surprised to say a wandering wanting,
in a writing rather feet in her arm, with more shaple dish fished away then before it was back to the players--the execution.' And he do please through the words quite thought Alice, who up to herself in the zitt of nearer.

Alice was sitting on the second to get up!
```
