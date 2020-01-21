from pyhive.hive import Cursor
from pyhive.hive import Connection as cnx
from pyhive.exc import ProgrammingError
from TCLIService.ttypes import TOperationState

# For visualisation
import seaborn
import matplotlib.pyplot as plt
from texttable import Texttable

# For caching
import datetime
import hashlib
import operator
import os
import pickle
import re
import time

# For async, logging
import requests
import logging

# For analysis
import pandas as pd
import pandas_profiling

from typing import List, Optional
from mypy_extensions import TypedDict

pformatDict = TypedDict('pformatDict', {'table': str, 'rowcount': int})


def connect(*args, **kwargs):
    return Connection(*args, **kwargs)


class CacheNotExistingError(Exception):
    pass


class NoResultSetError(ProgrammingError):
    def __init__(self):
        # The message mimicks what pyhive would do.
        super().__init__("No result set")


class FetchWithoutExecuteError(ProgrammingError):
    def __init__(self):
        # The message mimicks what pyhive would do.
        super().__init__("No query yet")


class TableFormattingError(Exception):
    pass


class UnknownPlotTypeError(Exception):
    pass


class NoTimelimeServerInfoError(Exception):
    pass


class Connection(cnx):
    def __init__(self, *args, **kwargs):
        self._phos_host = None
        self._phos_port = 10000  # default from pyhive

        # Host/port could be given via positional or kw arguments.
        # Technically it could be possible to pass an already initialised Thrift object, but then we're out of luck.
        if len(args):
            self._phos_host = args[0]
            if len(args) >= 2:
                self._phos_port = args[1]
        if 'host' in kwargs:
            self._phos_host = kwargs['host']
        if 'port' in kwargs:
            self._phos_host = kwargs['port']

        super().__init__(*args, **kwargs)

    def get_phos_host(self):
        return self._phos_host

    def get_phos_port(self):
        return self._phos_port


class Phos(Cursor):

    # On async execute, Yarn Timeline Server to get execution progress
    yts_url = None
    rm_url = None
    _host: Optional[str] = None
    _port: Optional[int] = None

    # Will be filled during setup()
    q: Optional['Query'] = None

    def __init__(self, *args, **kwargs):
        """
        To use Phos, instead of having `cnx=hive.connect(); cnx.cursor();` do `cnx=hive.connect(); phos(cnx);`
        """
        super().__init__(*args, **kwargs)
        if len(args):
            # args[0] is a connection object

            try:
                self._host = args[0].get_phos_host()
                self._port = args[0].get_phos_port()
            except AttributeError:
                # Not a Phos connection object. Too bad.
                pass
        logging.getLogger("pyhive.hive").setLevel(logging.WARN)
        logging.getLogger("requests.packages.urllib3.connectionpool").setLevel(logging.WARN)

    # Overridden methods

    def execute(self, query: str, **kwargs):
        # A bunch of things that will be needed for cache
        self.setup(
            query,
            docache=kwargs.pop('_cache', False),
            recache=kwargs.pop('_recache', False)
        )

        assert(self.q is not None)  # To prevent error during type checking
        if self.q.cache == self.q.READ:
            return

        if '_progress' in kwargs:
            do_async_wait = kwargs.get('_progress', False)
            del kwargs['_progress']
            if do_async_wait:
                try:
                    start = datetime.datetime.utcnow()
                    r = self._async_wait(**kwargs)
                    self.q.duration = datetime.datetime.utcnow() - start
                    return r
                # The exception is badly named. It catches all SIGINT, whether they come from the keyboard ot not.
                except KeyboardInterrupt:
                    print("SIGINT received, killing the query, be patient.")
                    raise

        # Else no async wait, normal flow.
        start = datetime.datetime.utcnow()
        r = super().execute(query, **kwargs)
        self.q.duration = datetime.datetime.utcnow() - start
        return r

    def fetchall(self):
        """Fetchall, but reads from cache if present."""
        self.assert_exec()
        if self.q.cache == Query.NONE:
            return super().fetchall()
        else:
            if not self.q.cache_exists():
                self.q.write_cache(
                    {
                        'description': super().description,
                        'data': super().fetchall(),
                        'query': self.q.query,
                        'duration': self.q.duration,
                        'ts_utc': datetime.datetime.utcnow(),
                    }
                )

            # Else data was already cached.
            # Todo: there might be cases where NoResultSetError should be raised.
            return self.q.read_cache()['data']

    @property
    def description(self):
        """Description, but reads from cache if present."""
        self.assert_exec()

        desc = None
        if self.q.cache == Query.NONE:
            # Cache not used
            desc = super().description
        else:
            if self.q.cache_exists():
                desc = self.q.read_cache()['description']
            else:
                # Data not yet cached
                desc = super().description

        if desc is None:
            raise NoResultSetError
        else:
            return desc

    # Internal methods

    def _async_wait(self, **kwargs):
        """Asynchronous execution, with progress display, CTR+C'able. """
        # Reset first query ID. It will be found from the logs below.
        self.query_id = None
        r = super().execute(self.q.query, async_=True, **kwargs)

        # Thrift statuses, with human names.
        statuses = TOperationState._VALUES_TO_NAMES

        full_error = []
        # Start marker of the useful error message
        short_prefix = 'info=['
        short_error = None

        status = None
        full_status_interval = 5
        last_full_status = datetime.datetime.utcnow()
        while status in [
            None,
            TOperationState.INITIALIZED_STATE,
            TOperationState.PENDING_STATE,
            TOperationState.RUNNING_STATE
        ]:

            time.sleep(0.5)
            new_status = self.poll().operationState

            if new_status != status:
                if status is not None:
                    # Keep the last line of status, which was written with \r
                    print()
                print(f"Status change. Was {statuses[status] if status is not None else 'None'}," +
                      f" is now {statuses[new_status]}.")
            status = new_status

            logs = self.fetch_logs()
            for message in logs:
                # Need to extract the query ID to talk to yarn
                if self.query_id is None:
                    # The id is displayed many times, in a few different formats.
                    # It looks like the earliest and most repeated is eg.
                    # (queryId=hive_20190628102710_bf894b75-f6d4-4da2-b0a5-2a2d44045711)
                    m = re.search(r'\(queryId=(?P<qid>hive.*?)\)', message)
                    if m:
                        self.query_id = m.group('qid')

                # If ERROR begins a line, let's remember and all after the full error message (contains massive
                # useless stacktrace and all attempts).
                # Extract as well the one relevant human-friendly line.
                if message.strip().startswith('ERROR') or (full_error and not message.strip().startswith('INFO')):
                    full_error.append(message)
                    if short_prefix in message and not short_error:
                        short_error = message.partition(short_prefix)[2]
                # Sometimes the error is only one line long, without error_prefix
                if not short_error and full_error:
                    short_error = full_error[0]

                logging.debug(message)
            if last_full_status + datetime.timedelta(seconds=full_status_interval) < datetime.datetime.utcnow():
                last_full_status = datetime.datetime.utcnow()
                try:
                    self._print_progress_info()
                except Exception as e:
                    # Whatever happens, trying to display progress should never stop the actual query run.
                    # Furthermore, some of those errors are transient (mostly at query start)
                    print("Error fetching progress info (query is probably not actually started yet): " + str(e),
                        end='\r')
        if full_error:
            self.q.full_error('\n'.join(full_error))
            self.q.short_error(short_error)
            print(self.q.short_error())

        print(f"Final status is {statuses[status]}.")
        return r

    def _print_progress_info(self):
        """
        Will print a line with interesting info (time, progress, some memory and cpu info).
        See https://hadoop.apache.org/docs/current/hadoop-mapreduce-client/hadoop-mapreduce-client-core/MapredAppMasterRest.html
        """
        if self.yts_url is None or self.rm_url is None:
            self._find_hadoop_urls()
        # 1) find application ID form the hive ID
        yts_query = (
            self.yts_url +
            # api path
            '/ws/v1/timeline/TEZ_DAG_ID?' +
            # If there are more than 1 result, I would not know what to do with it.
            'limit=2' +
            f'&primaryFilter=callerId:"{self.query_id}"' +
            # cache buster
            f'&_={int(datetime.datetime.utcnow().timestamp()*1000000)}'
        )
        logging.debug("Request to get applicationId : " + yts_query)
        yts_result = requests.get(yts_query).json()
        try:
            app_id = yts_result['entities'][0]['otherinfo']['applicationId']
        except (IndexError, KeyError) as e:
            raise NoTimelimeServerInfoError("No info in timeline server for query ID " + self.query_id)

        logging.debug(f"Application id : {app_id}")
        # 2) From the application, get the application-wide info.
        rm_query = (
            self.rm_url +
            # api path
            '/ws/v1/cluster/apps/' +
            app_id +
            # cache buster
            f'?_={int(datetime.datetime.utcnow().timestamp() * 1000000)}'
        )
        logging.debug(f"Resource manager url : {rm_query}")
        rm_result = requests.get(rm_query).json()['app']
        logging.debug(f"Resource manager results : {rm_result}")

        now = datetime.datetime.utcnow().timestamp()
        delta = 'Runtime: {:d}:{:02d}'.format(
            int((now - self.q.start)/60),
            int(now - self.q.start) % 60
        )

        print(
            ', '.join([
                f"Progress: {int(rm_result.get('progress', 0))}%",
                delta,
                f"Cluster: {int(rm_result.get('clusterUsagePercentage', 0))}%",
                f"Q: {int(rm_result.get('queueUsagePercentage', 0))}%",
                f"{rm_result.get('allocatedMB', 'n/a')} (+{rm_result.get('reservedMB', 'n/a')}) MB",
                f"{rm_result.get('allocatedVCores', 'n/a')} (+{rm_result.get('reservedVCores', 'n/a')}) cores",
                f"{rm_result.get('runningContainers', 'n/a')} containers",
            ]) + '.',
            # Magic: if end is \r, the next print will overwrite the current line.
            # TODO: it just prints over the previous line without erasing. If the previous line was longer,
            #  it does not look nice. Fix it.
            end='\r'
        )

    def _find_hadoop_urls(self):
        # From Hive 3 onwards, there is a web UI, there you we get information the relevant urls we need.
        # By default, it is on the hive server, port 10002.
        pass

    # Extra helpers
    @staticmethod
    def set_log_level(level: str):
        numeric_level = getattr(logging, level.upper(), None)
        logging.basicConfig(level=numeric_level)

    def setup(self, query: str, docache=False, recache=False) -> None:
        """
        Can be called before execute to get cache status of query.
        """
        self.q = Query(query, docache, recache)

    def del_cache(self):
        self.q.del_cache()

    def get_host(self) -> Optional[str]:
        return self._host

    def get_port(self) -> Optional[int]:
        return self._port

    @classmethod
    def set_yarn_ts_url(cls, yts_url: str) -> None:
        """Yarn timeline server url"""
        if re.match('http.?://', yts_url):
            cls.yts_url = yts_url
        else:
            cls.yts_url = f"http://{yts_url}"

    @classmethod
    def set_rm_url(cls, rm_url: str) -> None:
        """Resource Manager url"""
        if re.match('http.?://', rm_url):
            cls.rm_url = rm_url
        else:
            cls.rm_url = f"http://{rm_url}"

    def set_dag_name(self, name: str) -> None:
        """Note that there is no way to unset the dag name once set."""
        super().execute(f"set hive.query.name={name}")

    def assert_exec(self):
        if not self.q:
            raise FetchWithoutExecuteError

    def pformat(self, sort_by: List[int] = []) -> pformatDict:
        """Pretty format as a table. Uses fetchall()."""

        # Disabling max_width - output might be uglier if wide, but it will not be worse than beeline and will not
        # exception out.
        table = Texttable(max_width=0)
        # The only other decoration is the horizontal line between rows, which I do not want.
        table.set_deco(Texttable.BORDER | Texttable.HEADER | Texttable.VLINES)

        headers = self.headers()
        table.header(headers)

        # Convert some annoying types. Ints and friends will now not use the scientific notation anymore.
        # Note: the default will be 'a': auto.
        hive2table_types = {
            'bigint_type': 'i',
            'int_type': 'i',
            # Forcing to text prevents rounding and padding.
            'decimal_type': 't',
            'double_type': 't'
        }
        # self.description is a tuple with as 2nd element the data type.
        table.set_cols_dtype([hive2table_types.get(x[1].lower(), 'a') for x in self.description])

        cnt = 0
        try:
            for r in sorted(self.fetchall(), key=operator.itemgetter(*sort_by)) if sort_by else self.fetchall():
                cnt += 1
                table.add_row(r)
        except IndexError:
            raise TableFormattingError(f"You tried to sort by columns '{sort_by}' but at least one does not exist.")

        return {'table': table.draw(), 'rowcount': cnt}

    def pformat_extended(self, sort_by: List[int] = []):
        cnt = 0
        headers = self.headers()
        output = ''
        try:
            for r in sorted(self.fetchall(), key=operator.itemgetter(*sort_by)) if sort_by else self.fetchall():
                cnt += 1
                table = Texttable(max_width=0)
                table.set_deco(Texttable.VLINES)
                table.set_cols_align(['r', 'l'])
                table.set_chars(['-', ':', '+', '='])  # Just replacing vertical line with ':' instead of default '|'
                for i, h in enumerate(headers):
                    table.add_row([h, r[i]])
                output += f'row {cnt}:\n'
                output += table.draw()
                output += '\n\n'

        except IndexError:
            raise TableFormattingError(f"You tried to sort by columns '{sort_by}' but at least one does not exist.")

        return {'table': output, 'rowcount': cnt}

    def pprint(self, rowcount=True, sort_by: List[int] = [], extended=False) -> None:
        """Print the output of pformat()."""

        try:
            if extended:
                data = self.pformat_extended(sort_by)
            else:
                data = self.pformat(sort_by)
        except TableFormattingError as e:
            print(e)
            return

        print(data['table'])
        if rowcount:
            rc = f"{data['rowcount']} row{'s' if data['rowcount'] > 1 else ''}, "
            assert (self.q is not None)  # To prevent error during type checking
            if self.q.cache == Query.NONE:
                rc += 'cache skipped.'
            elif self.q.cache == Query.READ:
                cache = self.q.read_cache()
                write_ts = cache['ts_utc']
                ago = datetime.datetime.utcnow() - write_ts

                rc += 'read from cache, written on {w} utc ({ago}) ago. Original query duration: {duration}.'.format(
                    w=write_ts.replace(microsecond=0),
                    ago=ago - datetime.timedelta(microseconds=ago.microseconds),
                    duration=cache['duration'] - datetime.timedelta(microseconds=cache['duration'].microseconds)
                )

            elif self.q.cache == Query.WRITTEN:
                rc += 'written to cache.'
                cache = self.q.read_cache()
                rc += ' Query duration: {}.'.format(
                    cache['duration'] - datetime.timedelta(microseconds=cache['duration'].microseconds)
                )
            else:
                rc += 'but I have no idea if it was read from cache or not. That is weird.'
            print(rc)

    def headers(self):
        return [x[0] for x in self.description]

    def plot_xyh(self, plot='line', _x=0, _y=1, _hue=None, **kwargs) -> None:
        """Plot just about anything accepting x, y and optionally hue params. Uses fetchall()."""

        plot_call = getattr(seaborn, plot + 'plot', None)
        if not plot_call:
            raise UnknownPlotTypeError(f"'{plot}' is not a known plot type in seaborn.")

        # Seaborn needs column names, not index
        _x = self.headers()[_x]
        _y = self.headers()[_y]
        _hue = self.headers()[_hue] if _hue is not None else None

        df = self.get_df()
        plot_call(x=_x, y=_y, hue=_hue, data=df, **kwargs)
        plt.show()

    def get_df(self) -> pd.DataFrame:
        """returns a panda dataframe with headers. Uses fetchall()."""
        return pd.DataFrame(self.fetchall(), columns=self.headers())


class Query:

    # Cache status constants. If self.cache is None, then cache is not used at all.
    READ = 'read'
    WRITTEN = 'written'
    NONE = 'disabled'

    query: str
    hash: str
    cache_path: Optional[str] = None
    cache: str
    duration: datetime.timedelta

    _full_error: Optional[str] = None
    _short_error: Optional[str] = None

    # start : str

    def __init__(self, q: str, docache: bool = False, recache: bool = False):

        home = os.environ.get('HOME', None)
        if home is None:
            print("No HOME found, cannot cache.")
            docache = False
        else:
            # Following freedesktop specs: https://standards.freedesktop.org/basedir-spec/basedir-spec-latest.html
            self.cache_dir = os.environ.get('XDG_CACHE_HOME', home + "/.cache") + '/phos'

        self._set_props(q)

        if docache:
            if recache:
                logging.debug("Deleting cache")
                self.del_cache()

            if self.cache_exists():
                logging.debug("Cache exists.")
                self.cache = self.READ
            else:
                logging.debug("Cache Does not exist.")
                self.cache = self.WRITTEN
        else:
            logging.debug("Caching skipped.")
            self.cache = self.NONE

    def full_error(self, error: Optional[str] = None) -> Optional[str]:
        """Get/set full error string after execution."""
        if error:
            self._full_error = error
        else:
            return self._full_error

    def short_error(self, error: Optional[str] = None) -> Optional[str]:
        """Get/set short error string after execution."""
        if error:
            self._short_error = error
        else:
            return self._short_error

    def del_cache(self) -> None:
        """Delete cache, if it exists."""
        if self.cache_path is None:
            raise CacheNotExistingError()

        try:
            os.remove(self.cache_path)
        except FileNotFoundError:
            # Deleting a non-existing cache fails. Fair enough.
            pass

    def _set_props(self, query: str) -> None:
        self.query = query
        self.hash = self._query_hash()
        if self.cache_dir is not None:
            self.cache_path = os.path.join(self.cache_dir, self.hash)
        self.cache = self.NONE  # Might be overridden after this method
        self.start = datetime.datetime.utcnow().timestamp()

    def _query_hash(self) -> str:
        return hashlib.sha3_256(self.query.encode('utf-8')).hexdigest()

    def cache_exists(self) -> bool:
        """Assuming cache is in one file, which is the case with pickle."""
        if self.cache_path is None:
            raise CacheNotExistingError()

        # If size is 0, writing cache went wrong.
        return os.path.exists(self.cache_path) and os.stat(self.cache_path).st_size > 0

    def write_cache(self, data) -> None:
        if self.cache_path is None:
            raise CacheNotExistingError()

        # Let's be safe, and remove old cache first just in case.
        self.del_cache()

        # Make sure the cache directory exists.
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(self.cache_path, 'wb') as f:
            logging.debug("Dumping cache for the first time...")
            pickle.dump(data, f)
            logging.debug("... cache written.")
        if os.stat(self.cache_path).st_size == 0:
            logging.error('Cache dump failed (cache size 0) without known reason.')

    def read_cache(self) -> dict:
        if self.cache_path is None:
            raise CacheNotExistingError()

        with open(self.cache_path, 'rb') as f:
            return pickle.load(f)

    def __repr__(self) -> str:
        res = f"{type(self).__name__}: "
        res += ', '.join([f'{x}: {getattr(self, x)}' for x in ['cache', 'cache_path']])
        res += ', query='+self.query[:20]

        return res
