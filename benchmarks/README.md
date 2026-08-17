# Benchmarks

`scaling.py` measures build time for the new engine across a range of NPA and
prepare-and-measure hierarchies:

```bash
python benchmarks/scaling.py
```

To compare against MoMPy 1.x, install it into a separate environment and time
the same scenarios. Be warned that the 1.x build is superlinear: the
`PAM dimension nX=4` case (a 137x137 matrix) takes about 8.8 minutes there
versus 0.18 s here, so run the old one in the background.
