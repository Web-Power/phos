from prompt_toolkit import PromptSession
# Basic SQL - only ANSI
from pygments.lexers.sql import SqlLexer
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

from phos import Phos, Connection, NoResultSetError, UnknownPlotTypeError

from typing import List, Optional
import pyhive
# Look into fuzzy completers, progress bar

import csv
import re
import logging


class UnknownCommandError(Exception):
    pass


class Cli:

    # List of available commands, accessible with '!'.
    COMMANDS = {
        'bust': {
            'doc': 'Deletes cache of previous query.'
        },
        'cache': {
            'doc': 'Toggles cache.'
        },
        'csv': {
            'doc': 'Reruns the last query and save its output in a csv file. Takes one parameter: the csv file path/name.'
        },
        'error': {
            'doc': 'The error output is trimmed to remove massive stacktraces. This command shows the complete error.'
        },
        'explain': {
            'doc': 'Explains the previous query.'
        },
        'help': {
            'doc': 'Displays global help or of a command.'
        },
        'loglevel': {
            'doc': 'Sets loglevel.'
        },
        'ln': {
            'doc': 'Switches line numbers on and off when editing a multi-line query.'
        },
        'plot': {
            'doc': ('Plots a result set. Takes as first parameter a seaborn plot type. '
                    'All plot type with as parameter x, y and hue are valid (eg. line, bar). '
                    'x and y are mandatory parameters, the indices of the columns to plot. The 3rd parameter, hue, '
                    'is a sort of "group by". eg: "!plot line 1 3 2".'
                    )
        },
        'profile': {
            'doc': 'Profiles latest resultset, in html'
        },
        'py': {
            'doc': 'Switches to a python interpreter.'
        },
        'pydesc': {
            'doc': 'Shows description tuple from the python DB API.'
        },
        'recache': {
            'doc': 'Deletes the cache of the next query and recaches the execution result.'
        },
        'reuse': {
            'doc': 'Generates SQL to reuse the previous resultset as a CTE (mostly useful if previous query was cached).'
        },
        'sortby': {
            'doc': ('Rerun the last query and sort it. Uses cache if present. '
                    'Takes as parameter a space separated list of column indexes to sort by.')
        },
        'x': {
            'doc': 'Toggle extended (one row per line)/table output.'
        },
    }

    def __init__(self, cur):
        # File needs to exist!
        # Todo: use good defaults
        self.session = PromptSession(history=FileHistory('/home/guillaume/.pyline'))
        self.db = 'default'
        self.cur = cur
        self.show_line_numbers = False
        self.extended_display = False
        self.log_level = logging.getLevelName(logging.getLogger().getEffectiveLevel())

        self.use_cache = True

        # Optional prefilled command
        self.default_line: str = ''

        # Handle '!' commands
        self.command = {}

        self.search_use = re.compile(r'^use\s+(?P<db>[^\s]*?)\s*$', re.DOTALL | re.MULTILINE | re.IGNORECASE)
        self.search_command = re.compile(r'!(?P<command>\w+)(\s+(?P<params>.*?))?\s*$')

    def input_loop(self):
        print("Enter !help for help.")
        print("Sending a command is done with Meta+Enter (on Linux Meta = Alt).")
        while True:
            li = ''
            try:
                bottom = [
                    f'Connected to {self.cur.get_host()}:{self.cur.get_port()}',
                    f'Caching is {"on" if self.use_cache else "off"}.',
                    f'Log level is {self.log_level}',
                    f'Extended display is {"on" if self.extended_display else "off"}.'
                ]

                li = self.session.prompt(
                    self.db + '> ',
                    default=self.default_line,
                    lexer=PygmentsLexer(SqlLexer),
                    auto_suggest=AutoSuggestFromHistory(),
                    bottom_toolbar=' - '.join(bottom),
                    multiline=True,
                    prompt_continuation=self._prompt_continuation,
                    # mouse_support=True
                )
            except KeyboardInterrupt:  # CTRL-C
                continue
            except EOFError:  # CTRL-D
                break
            finally:
                # Whatever happens, reset default prefilled command.
                self.default_line = ''

            try:
                self.exec(li.strip())
            except Exception as e:
                logging.error(e)
                raise

        print('Bye.')

    def exec(self, l: str):
        for q in self._split_query(l):
            command = self.handle_command(l)
            if command is None:
                # Some commands are done after executing (eg. !help), some require carrying on (!explain).
                return

            # If the query is actually sortby, then we rerun it. It will come from cache and it will be sorted
            if command == 'sortby':
                q = self.cur.q.query
            if command == 'explain':
                q = 'explain ' + self.cur.q.query

            # Sending empty string to execute does not go well.
            # Todo: sending only comments does not go well either (and beeline can handle it).
            if not q:
                continue

            self.find_and_set_db(q)
            recache = False
            cache = self.may_cache(q) and self.use_cache
            if self.command.get('command', '') == self.COMMANDS['recache']:
                recache = True

            try:
                self.cur.execute(q, _cache=cache, _recache=recache, _progress=True)
                self.cur.pprint(sort_by=self.command.get('sortby', []), extended=self.extended_display)
                # Reset last command.
                self.command = {}
            except pyhive.exc.OperationalError as e:
                self.cur.q.del_cache()
                # Todo when debugging add sqlState,statusCode
                print(e.args[0].status.errorMessage)
            except pyhive.exc.ProgrammingError as e:
                self.cur.q.del_cache()
                if(str(e).lower()) != 'no result set':
                    raise
                # else not resultset, fair enough.
            except NoResultSetError:
                # Phos caught that one itself.
                pass
            except KeyboardInterrupt:
                print("Query killed.")
                # query was killed.

    @staticmethod
    def may_cache(q: str) -> bool:
        """
        Some statements should really not be cached (use, set, desc, explain, DDLs).

        Note that insert/updates inside CTE are not found here, but as they will have no resultset, any caching attempt
        will roll back and have no effect.
        """
        no_cache = ['abort', 'add', 'alter', 'analyze', 'commit', 'create', 'desc', 'drop', 'explain', 'insert',
                    'set', 'show', 'start', 'rollback', 'update', 'use']
        lower_q = q.lower()
        for kw in no_cache:
            if lower_q.startswith(kw):
                return False
        return True

    def find_and_set_db(self, q: str) -> None:
        g = self.search_use.search(q)
        if g:
            self.db = g.group('db')

    def handle_command(self, l: str) -> Optional[str]:
        """Handle !commands. Return None is a command was handled and execution should stop, return command name
        (possibly empty) otherwise."""
        match = self.search_command.match(l)
        if not match:
            # Not a command
            return ''

        command = match.group('command')
        params = match.group('params')
        canon = self._find_command(command)
        if canon is None:
            print(f"Command {command} does not exist. Use !help for help.")
            return None

        # From there on, we know that there is a command.
        if canon == 'sortby':
            # sortby expects a space separated list of int
            if params:
                try:
                    params = [int(x) for x in params.split()]
                except ValueError:
                    print(f"'!{canon}' requires a space separated list of ints. Got {params}.")
                    return None
            else:
                print(f"'!{canon}' requires a space separated list of ints.")

            self.command = {
                'command': command,
                'sortby': params
            }
            # Exec needs to continue
            return canon
        elif canon == 'csv':
            # csv expect a file name
            if not params:
                logging.error(f"{canon} require a csp ptah as parameter. Got none.")
            else:
                outcsv = params
                try:
                    # newline mandatory for csvwriter
                    with open(outcsv, mode='w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(self.cur.headers())
                        for r in self.cur.fetchall():
                            writer.writerow(r)
                except OSError as e:
                    logging.error(f'Could not open {outcsv} for writing: {e}')
                r = len(self.cur.fetchall())
                print(f"{r} row{'s' if r>1 else ''} written to {outcsv}.")
            return None
        elif canon == 'plot':
            # expects a plot type and 2 or 3 column ids
            if params:
                pg = params.split()
                try:
                    plot = pg[0]
                    indices = [int(x) for x in pg[1:]]
                except ValueError:
                    logging.error(f"'!{canon}' requires a plot type and a space separated list of ints. Got {params}.")
                    return None

                num_cols = len(indices)
                if num_cols not in (2, 3):
                    print(f"'!{canon}' requires 2 or 3 ints. Got {num_cols}.")
                    return None

                try:
                    self.cur.plot_xyh(
                        plot=plot, _x=indices[0], _y=indices[1], _hue=indices[2] if num_cols == 3 else None
                    )
                except UnknownPlotTypeError as e:
                    logging.error(e)
            else:
                logging.error(f"'!{canon}' requires a plot type and a space separated list of ints. Got {params}.")
            return None
        elif canon == 'py':
            from ptpython.repl import embed
            df = self.cur.get_df()

            def config(python_repl):
                python_repl.confirm_exit = False

            # Preload imports used for Python repl
            import seaborn as sns
            import matplotlib.pyplot as plt
            import pandas as pd
            import numpy as np

            print(
                "You are switching to a python interpreter. The variable `df` holds the data of your "
                "last query in a pandas dataframe. Seaborn (sns), maptplotlib (plt), pandas (pd) and numpy (np)"
                "are already loaded."
            )
            embed(globals(), locals(), configure=config)
            print("You switched back to the sql interpreter.")
            return None
        elif canon == 'bust':
            self.cur.q.del_cache()
            print("Cache of previous query cleared.")
            return None
        elif command == 'help':
            if params:
                print(self.COMMANDS[params]['doc'])
            else:
                self._print_help()
            return None
        elif canon == 'ln':
            self.show_line_numbers = not self.show_line_numbers
            return None
        elif canon == 'pydesc':
            print(self.cur.description)
            return None
        elif canon == 'explain':
            # This command will just append 'explain' to the actual query, when it is executed.
            return canon
        elif canon == 'error':
            print(self.cur.q.full_error())
            return None
        elif canon == 'x':
            # This command will just change the display
            self.extended_display = not self.extended_display
            print(f"Extended display {'on' if self.extended_display else 'off'}.")
            return None
        elif canon == 'cache':
            # This command will just change the display
            self.use_cache = not self.use_cache
            print(f"Cache {'on' if self.use_cache else 'off'}.")
            return None
        elif canon == 'loglevel':
            if params:
                level = params.upper()
                numeric_level = getattr(logging, level, None)
                if numeric_level is None:
                    logging.error(f"Loglevel {level} invalid. Keeping current level.")
                else:
                    self.set_log_level(level)
                    self.cur.set_log_level(level)
            else:
                logging.error(f"{canon} requires log level as paraemter. Got nothing.")
            return None
        elif canon == 'reuse':
            if self.cur.q is None:
                print("First query of the session, cannot reuse previous result set.")
            elif self.cur.q.cache_exists():
                cnt = 0
                data = []
                # description is a tuple with as 2nd element the data type, of the form INTEGER_TYPE.
                # Make it an easy lowercase list.
                types = [x[1].lower().replace('_type', '') for x in self.cur.description]
                headers = self.cur.headers()
                for datarow in self.cur.fetchall():
                    cnt += 1
                    valuerow = []
                    for idx, cell in enumerate(datarow):
                        if types[idx] in ['bigint', 'int', 'decimal']:
                            # Just put the cell, no quotes, but cast to get right type in case of null.
                            valuerow.append(str(cell) if cell is not None else f'cast(null as {types[idx]})')
                        elif types[idx] in ['double']:
                            # Hardcoded doubles are found as decimal.
                            valuerow.append(f'cast({"null" if cell is None else str(cell)} as {types[idx]})')
                            # valuerow.append(str(cell) if cell is not None else f'cast(null as {types[idx]})')
                        elif types[idx] in ['string']:
                            # Needs quotes.
                            valuerow.append(f"'{cell}'" if cell is not None else 'cast(null as string)')
                        elif types[idx] in ['timestamp']:
                            # Needs cast.
                            valuerow.append(
                                "cast(" +
                                (f"'{cell}'" if cell is not None else 'null') +
                                " as timestamp)"
                            )
                        elif types[idx] in ['null']:
                            valuerow.append('null')
                        else:
                            logging.error(
                                f"Column {headers[idx]} is of type '{types[idx]}'"
                                " which I do not know how to handle.")
                            return None
                    data.append(', '.join(valuerow))

                if cnt == 0:
                    print("Previous query had no or an empty result set, cannot reuse.")
                else:
                    cte = "with previous as (\n"
                    cte += "  select lv.* from (select 0) t\n"
                    cte += "  lateral view stack(\n"
                    cte += f"    {cnt},\n    "
                    cte += ",\n    ".join(data)
                    cte += "\n  ) lv as `" + '`, `'.join(headers) + "`\n"
                    cte += ")\n"
                    cte += "select * from previous"

                    self.default_line = cte
            else:
                print("Previous query not cached, cannot reuse previous result set.")
            return None
        elif canon == 'profile':
            print("Generating the result set profile. It might take a while.")
            import webbrowser
            from pathlib import Path

            df = self.cur.get_df()
            profile = df.profile_report(title="Query resultset profiling")
            path = Path.cwd().joinpath(Path("output.html"))
            profile.to_file(output_file=path)
            webbrowser.open_new_tab(path.as_uri())

            print(f"{path.as_uri()} has been opened in your browser.")
            return None
        elif canon == 'recache':
            # We need to remember that one. It will have effect only later.
            self.command = {
                'command': 'recache'
            }
            return None

        # Should never be reached as every branch should have their own return statement.
        logging.error(f"Unhandled command '{command}'. Report bug.")
        return None

    def _find_command(self, c):
        """
            Returns canonical name for a command. Take care of case.
            Todo: accept shortest possible command prefix
        """
        canons = [k.lower() for k in self.COMMANDS.keys()]
        command = c.lower()
        if command in canons:
            return c
        else:
            return None

    def _print_help(self):
        # TODO: expand
        print(self.COMMANDS)

    def _prompt_continuation(self, width, line_number, is_soft_wrap):
        # line_number is for continuation only, so starts as 2. 1 is actually the initial continuation line.
        # Adding + 1 keeps it in sync with what Hive thinks.
        ln = str(line_number + 1) if self.show_line_numbers else ''
        return ln + ' ' * (width - len(ln))

    def set_log_level(self, level: str):
        self.log_level = level.upper()
        numeric_level = getattr(logging, self.log_level, None)
        logging.basicConfig(level=numeric_level)

    @staticmethod
    def _split_query(q: str) -> List[str]:
        # todo: not split if between quotes
        # All queries to be returned
        all_qs = []
        # Current query being read. This will be an array of lines.
        current_q = []
        # This stack could be changed while looping, if for instance a semicolon is found (then half a line is
        # processed, the rest will be put back on the stack).
        stack = q.split('\n')
        # Reversing the lines to use pop() and append() without needing an actual collection.deque.
        stack.reverse()

        while True:
            try:
                li = stack.pop()
            except IndexError:
                break

            # Reminder: partition() returns 3 strings: the one before the partitioner, the partitioner itself and the
            # rest. If the partitioner is not found, the 2nd and 3rd string are empty (not None).
            cmd, dbldash, comment = li.partition('--')
            cmd1, semicolon, cmd2 = cmd.partition(';')

            # A ';' appears outside a comment
            if semicolon:
                current_q.append(cmd1)
                # Cmd1 1 is complete.
                all_qs.append(current_q)
                # Put the rest (cmd2) back on the stack to process it and reset current accumulator.
                current_q = []
                stack.append(cmd2 + dbldash + comment)
            else:
                # Nothing exciting, move on.
                current_q.append(li)

        all_qs.append(current_q)
        return ['\n'.join(x).strip() for x in all_qs]


if __name__ == "__main__":
    cnx = Connection('hive-server-host')
    c = Phos(cnx)
    c.set_yarn_ts_url('timeline-server-host:8188')
    c.set_rm_url('resource-manager-host:8088')

    cli = Cli(c)
    cli.input_loop()
