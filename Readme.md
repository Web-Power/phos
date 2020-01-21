Phos: PyHive On Steroids

# Note

This project is a (quite advanced) WIP. I am leaving my current job without having time to polish it. I am (was) using it daily, so it's pretty good as far as I am concerned.

The todos I would have wanted to reach are:

- installable via pypi/conda
- better start script
- better configurability (eg. cache location)
- startup options (eg. caching on or off, hive host...)
- better readme
- more tests

Hopefully I will be able to carry on, otherwise it's open source (MIT license), so feel free to contribute or fork.

# Rationale

This project scratches an itch. I wanted to be able to run hive queries via Python, while seeing progress.

# Usage

## Driver

Phos only extends the hive cursor from PyHive. It thus is fully compatible with current pyhive. The
behaviour changes slightly (and for the better, of course) only if you use the `_progress` or `_cache`
parameters to `execute`.

To use Phos, create the hive/phos cursor passing the connection object, instead of calling the `cursor` method
from the connection object.

```python
cnx=hive.connect()
phos(cnx)

# Instead of the other option:
# cnx=hive.connect()
# cnx.cursor()
```


## Cli (beeline replacement)

`PYTHONPATH=. ./env/bin/python ./phos/cli.py`

This will give you a sql prompt, to connect to hive. Compared to beeline, it features:

- very fast startup
- display of current DB in the prompt
- multi-line edit
- coloration
- (optional) caching of results
- reuse of resultsets
- graphing of resultset
- switch to a python shell with the last resultset in a panda dataframe
- profiling of the last resultset

# Features
## Caching

The first use case it to extend `cursor` class to be able to cache query results.

If the `_cache=True` parameter is passed to an `execute` call, then
- if `_recache=True` is passed, potential old cache will be deleted first,
- if the data is already cached, `execute` will not run at all,
- when `fetchall` is called for the first time, it will cache (pickle on disk) the data, and read the cache on
subsequent calls.

The cache is quite basic for now:
- Only `fetchall` caches,
- it puts data inside a hard-coded directory,
- the cache id is the sha3-256 of the query string.

## Progress display

The second use case is to see progress of queries. For now, only some details are printed. If you pass
the parameter `_progress=True` to `execute`, you will see the following lines:
```
Runtime: 0:12, progress: 3%, Q: 285%, Cluster: 85%, VCores: 38, Containers: 38, Reserved Cores: 2, Mem: 298.0GB, Reserved: 16.0GB
Runtime: 0:14, progress: 4%, Q: 285%, Cluster: 85%, VCores: 38, Containers: 38, Reserved Cores: 2, Mem: 298.0GB, Reserved: 16.0GB
Runtime: 0:20, progress: 8%, Q: 277%, Cluster: 83%, VCores: 37, Containers: 37, Reserved Cores: 3, Mem: 290.0GB, Reserved: 24.0GB
Runtime: 0:26, progress: 13%, Q: 292%, Cluster: 87%, VCores: 39, Containers: 39, Reserved Cores: 2, Mem: 306.0GB, Reserved: 16.0GB
...
Final status is FINISHED_STATE.
```

## Visualisations

The third use case is easy use of the results. This works best with cache, as it relies on `fetchall`, and
multiple uncached `fetchall` are not possible.

- plotting of result. Some `plot` functions will be added, when I need them.
- there is as well a `pp` (pretty print) method, to display a table output on the terminal.
- `get_df` will give you a pandas dataframe.

# Caveats

None, my code is obviously perfect and smell of roses.

That said, the environment built is big (2G) because it has a lot of statisticy graphiquy stuff.

Note that all the visualisation methods make use of `fetchall`, which is not repeatable by default (a second call to
`fetchall` would return an empty set as the full data has been consumed already). They are meant to only
work with the cache, currently.

As it was mostly an itch scratching project, there are not many tests (yet).

# Todo

Everything is printed with `print`, it might be nicer to use a logger, with options.

There is no task progress, only full query progress.


