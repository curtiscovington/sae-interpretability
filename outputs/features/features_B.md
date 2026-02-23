# Feature summaries (B)

## Feature 687 — none, filename, len_rv
- mean: 0.8079
- max: 2.6048
- frequency: 0.9776
  - (rv, tuple):             len_rv = len(rv)# a 3-tuple is
  -  " + encoding             else:                 msg = "unknown encoding for {!r}: {}".
  -  not isinstance(indent, str):         indent = ' ' * indent     return _format(
  - offset             else:                 col_offset = node.col_offset         if 'end
  - _col_offset].decode()if padded:         padding = _pad_whitespace

## Feature 479 — import, from, werkzeug.exceptions
- mean: 0.7826
- max: 2.4631
- frequency: 0.9787
  - ] is e:                 raiseraise eself.log_exception(ctx, exc_info)
  -  uses a         308 status which tells the browser to resend the method and         body...
  - _hosts := self.config["TRUSTED_HOSTS"]) is not None:                 request
  -  the coroutine function... code-block:: pythonresult = app.async_to_sync
  -  shell context         processors... versionadded:: 0.11         """         rv =

## Feature 116 — used, this, with
- mean: 0.7824
- max: 2.2156
- frequency: 0.9784
  -      The detect_encoding() function is used to detect the encoding that should     be used to
  -  :attr:`debug` flag is set the server will automatically reload         for code changes and show a
  -  uses a         308 status which tells the browser to resend the method and         body...
  - .         :param values: Values to use for the variable parts of the URL             rule.
  - print_name is not None:                     endpoint = f"{blueprint_name}{endpoint}"

## Feature 17 — import, elif, error
- mean: 0.7828
- max: 2.4186
- frequency: 0.9750
  - , list)):                     rv, headers = rv  # pyright: ignore                 else: 
  -  proxies in there.             request=request,             session=session,             g=
  - elif isinstance(value, AST):                 self.visit(value)class NodeTransformer(Node
  - rv, (dict, list)):                 rv = self.json.response(rv)             
  -  else.In debug mode, intercept a routing redirect and replace it with         an error if the body

## Feature 611 — none, raise, return
- mean: 0.7691
- max: 2.2853
- frequency: 0.9780
  - SERVER_NAME"] is not None:             return self.url_map.bind(                 
  - _DEFAULT"]if value is None:             return Noneif isinstance(value, timedelta): 
  -             token_type = token.type             if args.exact:                 token_type =
  - format(filename,                         encoding)             raise SyntaxError(msg)if bom_found
  -      if encoding is not None:         if encoding == "utf-8-sig": 

## Feature 329 — error, return, filename
- mean: 0.7565
- max: 2.3665
- frequency: 0.9698
  -  default, [first]second = read_or_stop()     if not second:         
  -          to how resources are looked up.  However it will make debugging more         painful.
  - '         return encodingfirst = read_or_stop()     if first.startswith
  - message)         sys.stderr.write('\n')def error(message, filename=None
  - utf-8'     def read_or_stop():         try:             return read

## Feature 741 — none, that, want
- mean: 0.7522
- max: 2.2856
- frequency: 0.9752
  - expr, re.UNICODE)# Note that since _all_string_prefixes includes the
  - information and a lot more.So it's important what you provide there.  If you are using a
  -  None,         _scheme: str | None = None,         _external: bool | None
  - ), in no specified order.  This is useful if you     only want to modify nodes in place
  -  context processors want         to inject.  Note that the as of Flask 0.6, the

## Feature 637 — else, return, encoding
- mean: 0.7393
- max: 2.5641
- frequency: 0.9758
  -             else:                 return True         else:             return type(a) is type(
  - _attr:                 return False         else:             return Trueif type(a) is
  - _field):                 return False         else:             return Truedef _compare_attributes(
  - , b_item in zip(a, b):                 if not _compare(a_item
  - external = _scheme is not None         else:             # If called by helpers.url

## Feature 42 — return, token, encoding
- mean: 0.7341
- max: 2.2392
- frequency: 0.9772
  -  all token lists start with an ENCODING token which tells you which encoding was used to decode the
  -             information... versionchanged:: 1.0             If installed, python-dotenv will be
  -  None,             "TESTING": False,             "PROPAGATE_EXCEPTIONS": None
  -  len(indent)                 startline = False             elif tok_type in {FSTRING_
  - Name(id='data', ctx=Load()),                    slice=Constant(value=node.id

## Feature 144 — param, group, first
- mean: 0.7337
- max: 2.4404
- frequency: 0.9769
  - Token = Ignore + PlainToken# First (or only) line of ' or " string. 
  - . The old signature is deprecated"                     " and will not be supported in Flask 4.0
  - ' +                 group('"', r'\\\r?\n')) PseudoExtras
  -  and end_col_offset are optional attributes, and they             # should be copied whether the value
  -              variables from :file:`.env` and :file:`.flaskenv` files.

## Feature 106 — return, this, useful
- mean: 0.7310
- max: 2.7354
- frequency: 0.9788
  - ]         rv = self.response_class()         rv.allow.update(methods)
  - .  This is useful if         you want to access the context locals for testing::with app.
  -  [first]second = read_or_stop()     if not second:         check(
  -  extensions to improve debugging         information and a lot more.So it's important what you provide there.
  -  = ctx.url_adapter.allowed_methods()  # type: ignore[union-attr]

## Feature 242 — import, from, return
- mean: 0.7297
- max: 2.2164
- frequency: 0.9772
  - Finally, the :data:`request_tearing_down` signal is sent.:param exc:
  -  collect_errors:             request_tearing_down.send(self, _async_wrapper
  - _info()[1]                 raise             return response(environ, start_response) 
  - _flashed_messages=get_flashed_messages,             config=self.config,
  - "}:             raise ValueError("Resources can only be opened for reading.")path = os.path.

## Feature 714 — type, value, none
- mean: 0.7308
- max: 2.0624
- frequency: 0.9737
  - ,                     headers=headers,  # type: ignore[arg-type]                 ) 
  - _argument(dest='filename', nargs='?',                         metavar='filename.py
  - )     parser.add_argument(dest='filename', nargs='?',                         met
  - startline = False             elif tok_type in {FSTRING_MIDDLE, T
  - globals.update(             url_for=self.url_for,             get_

## Feature 201 — param, this, raise
- mean: 0.7300
- max: 2.4779
- frequency: 0.9735
  - param _anchor: If given, append this as ``#anchor`` to the URL.         :
  -  methods of the app. Changing         :attr:`jinja_options` after this will have no
  - SCHEME`.         :param data: The request body text or bytes,or a dict of
  - 5000`` or the             port defined in the ``SERVER_NAME`` config variable if present. 
  - )     parser.add_argument(dest='filename', nargs='?',                         met

## Feature 195 — none, return, ignore
- mean: 0.7282
- max: 2.4724
- frequency: 0.9745
  - .request.blueprints)if handler is not None:             server_error = self.ensure
  - (before_func)()if rv is not None:                         return rv  # type: ignore
  -  self.config["SERVER_NAME"] is not None:             return self.url_map.
  -  or annotated AppContext             if param is None or not (                 # no annotation, match name
  - )         error: BaseException | None = None         try:             try:                 

## Feature 31 — none, open, headers
- mean: 0.7097
- max: 2.1871
- frequency: 0.9804
  -  or headers             elif len_rv == 2:                 if isinstance(rv[1], (
  - packed directly             if len_rv == 3:                 rv, status, headers = rv  
  -  at :attr:`static_url_path` if :attr:`static_folder` is         
  -  Request) -> t.NoReturn:         """Intercept routing exceptions and possibly do something else.
  -              elif toknum == DEDENT:                 indents.pop()                 continue

## Feature 458 — **kwargs, return, t.any
- mean: 0.7098
- max: 1.9358
- frequency: 0.9772
  - ', re.ASCII) blank_re = re.compile(br'^[ \t\
  - ): return group(*choices) + '?'# Note: we use unicode matching for names ("\
  - def wrapper(self: Flask, *args: t.Any, **kwargs: t.Any
  -  don't contain any permutations (include 'fr', but not     #  'rf'). The various
  -  value of the view or error handler.  This does not have to         be a response object.

## Feature 298 — none, import, encoding
- mean: 0.7064
- max: 2.2372
- frequency: 0.9757
  - in_open(filename, 'rb')     try:         encoding, lines = detect_
  - new_values.extend(value)                             continue                     new_values.append(value
  -     from .testing import FlaskCliRunner     from .typing import HeadersValueT_
  - """     ut = Untokenizer()     out = ut.untokenize(iterable
  - parse.ArgumentParser(color=True)     parser.add_argument('infile', n

## Feature 260 — none, bool, return
- mean: 0.7006
- max: 2.5314
- frequency: 0.9793
  -          # Ignore this call so that it doesn't start another server if         # the '
  -          _external: bool | None = None,         **values: t.Any, 
  -          _anchor: str | None = None,         _method: str | None = None
  -  None,         debug: bool | None = None,         load_dotenv: bool =
  - .request.blueprints)if handler is not None:             server_error = self.ensure

## Feature 577 — import, instead, **kwargs
- mean: 0.6948
- max: 2.0487
- frequency: 0.9761
  -  ast.Tuple instead."""     def __new__(cls, dims=(), **kwargs): 
  -  len(b):                 return False             for a_item, b_item in zip(
  -  index value directly instead."""     def __new__(cls, value, **kwargs):         return
  -  closing down         of the context until the end of the ``with`` block.  This is useful
  - _until_next_bracket = False         for character in token:             if character == "

## Feature 281 — none, return, string
- mean: 0.6909
- max: 2.1013
- frequency: 0.9777
  -          default = 'utf-8-sig'     if not first:         return default
  - argument(dest='filename', nargs='?',                         metavar='filename.py',
  - able):         it = iter(iterable)         indents = []         startline
  -  StringPrefix = group(*_all_string_prefixes())# Tail end of ' string
  -         rule: Rule = req.url_rule  # type: ignore[assignment]         

## Feature 501 — versionadded, with, error
- mean: 0.6917
- max: 2.0343
- frequency: 0.9761
  -  be opened         with:.. code-block:: pythonwith app.open_resource("schema.
  - foo``) to ``data['foo']``::class RewriteName(NodeTransformer):def
  -  the coroutine function... code-block:: pythonresult = app.async_to_sync
  -         self.cli = cli.AppGroup()# Set the name of the Click group in case
  -  Called by         :meth:`.AppContext.pop`.This calls all functions decorated with :meth

## Feature 499 — request, outside, self.prev_row
- mean: 0.6907
- max: 2.1882
- frequency: 0.9754
  - down` signal is sent.:param exc: An unhandled exception raised while the context was active
  - Error(                     "Unable to build URLs outside an active request"                     " without 'SERVER_
  - down` signal is sent.:param exc: An unhandled exception raised while dispatching the request
  - visit(ast_obj)def main(args=None):     import argparse     import
  -      attributes) from *old_node* to *new_node* if possible, and return

## Feature 383 — none, _valid_string_prefixes, func
- mean: 0.6914
- max: 2.4196
- frequency: 0.9733
  -  various permutations will be generated.     _valid_string_prefixes = ['b', 'r
  - """         cls = self.test_client_class         if cls is None: 
  -                  for func in self.template_context_processors[name]:                     context.update
  -  method in (             cls.handle_http_exception,             cls.handle_user_
  - e):             return self.handle_http_exception(ctx, e)handler = self._

## Feature 336 — self.prev_col, self.prev_row, node
- mean: 0.6926
- max: 2.3022
- frequency: 0.9703
  -  self.prev_row and col < self.prev_col:             raise ValueError("start
  - node, BinOp)         and isinstance(node.op, (Add, def dump(
  - .prev_col = 0         self.prev_type = None         self.prev_
  -  self.prev_row or row == self.prev_row and col < self.prev_col
  -  node.elts))     if isinstance(node, Set):         return set(map(_

## Feature 229 — return, from, testing
- mean: 0.6850
- max: 2.1585
- frequency: 0.9798
  -         # those unchanged as errors         if e.code is None:             return e# R
  - _ERRORS"]         ):             e.show_exception = Trueif isinstance(e,
  -  might be created while the server is running (e.g. during         # development). Also,
  - from .testing import EnvironBuilderbuilder = EnvironBuilder(self, *args, **kwargs)
  - _ROOT"],                 url_scheme=self.config["PREFERRED_URL_SC

## Feature 408 — response, tokval, sn_host
- mean: 0.6931
- max: 2.3787
- frequency: 0.9677
  - .:internal:         """         response = self.make_response(rv)         
  -  e:                 error = e                 response = self.handle_exception(ctx, e)
  - :                 ctx.push()                 response = self.full_dispatch_request(ctx)
  -              if sn_host:                 host = sn_host             else:                 host
  - _or_tstring:                 tokval = ' ' + tokval# Insert a space between

## Feature 361 — return, *args, **kwargs
- mean: 0.6853
- max: 2.3717
- frequency: 0.9770
  - ]') Number = group(Imagnumber, Floatnumber, Intnumber)# Return the empty
  - ):         major, minor = feature_version  # Should be a 2-tuple.         
  - ("authentication")                     super(CustomClient,self).__init__( *args, **kwargs
  - dims = property(_dims_getter, _dims_setter)class Suite(
  - .path.join(self.instance_path, resource)if "b" in mode: 

## Feature 697 — import, flask, this
- mean: 0.6855
- max: 2.5363
- frequency: 0.9751
  - :`Flask` instance in your main module or     in the :file:`__init__.py
  - response_class`.:func:`callable`                 The function is called as a WSGI application
  -  Untokenizer:def __init__(self):         self.tokens = []         self.
  - \\.[^\n"\\]*)*' +                 group('"', r'\\\r?
  - ` if :attr:`static_folder` is         set.Note this is a duplicate of the

## Feature 612 — elif, return, port
- mean: 0.6878
- max: 2.1973
- frequency: 0.9702
  -  self.json.response(rv)             elif isinstance(rv, BaseResponse) or callable
  - else:         padding = ''first = padding + lines[lineno].encode()[col_offset
  - .         :return: a new response object or the same, has to be an                  instance
  - 127.0.0.1"if port or port == 0:             port = int(
  - )                 status = headers = None             elif isinstance(rv, (dict, list)): 

## Feature 710 — none, application_root, max_content_length
- mean: 0.6835
- max: 2.0814
- frequency: 0.9748
  - _BAD_REQUEST_ERRORS": None,             "TRAP_HTTP_EXCEPTIONS":
  - _MEMORY_SIZE": 500_000,             "MAX_FORM_PARTS": 1
  -              "APPLICATION_ROOT": "/",             "SESSION_COOKIE_NAME":
  -             "SECRET_KEY": None,             "SECRET_KEY_FALL
  - FRESH_EACH_REQUEST": True,             "MAX_CONTENT_LENGTH": None

## Feature 0 — once, group(stringprefix, module
- mean: 0.6856
- max: 2.2211
- frequency: 0.9716
  - Warning,                     stacklevel=2,                 )                 setattr(cls, method.__
  -  the ``flask`` command         #: once the application has been discovered and blueprints have         
  -  group(StringPrefix + "'''", StringPrefix + '"""') # Single-line ' or
  -         an SQL query in debug mode.  If the import name is not properly set         up,
  - dims), Load(), **kwargs)# If the ast module is loaded more than once, only add

## Feature 125 — request, encoding, finalizing
- mean: 0.6801
- max: 2.0423
- frequency: 0.9793
  -  check(line, encoding):         # Check if the line matches the encoding.         if 0
  -  context functions.         orig_ctx = context.copy()for name in names:             if
  - self.ensure_sync(func)())context.update(orig_ctx)def make_shell
  -                  "Request finalizing failed with an error while handling an error"             )         return
  -  For one, it might be created while the server is running (e.g. during         #

## Feature 362 — none, request, url_adapter
- mean: 0.6814
- max: 2.4869
- frequency: 0.9736
  -  not None:                 url_adapter = ctx.url_adapter             else:                 url
  - _request:             url_adapter = ctx.url_adapter             blueprint_name =
  - _adapter             blueprint_name = ctx.request.blueprint# If the endpoint starts with
  - .7         """         methods = ctx.url_adapter.allowed_methods()  #
  - exception(self, request: Request) -> t.NoReturn:         """Intercept routing exceptions

## Feature 661 — none, name, _scheme
- mean: 0.6807
- max: 2.3260
- frequency: 0.9713
  - , *args, **kwargs):                     self._authentication = kwargs.pop("authentication
  -         self,         import_name: str,         static_url_path: str |
  -  None,         _scheme: str | None = None,         _external: bool | None
  -         """         rv = {"app": self, "g": g}         for processor in
  -         root_path: str | None = None,     ):         super().__init__(

## Feature 145 — **kwargs, node, line
- mean: 0.6775
- max: 2.0717
- frequency: 0.9755
  - node):     """     Used by `literal_eval` to convert an AST node into a
  - ='the file to parse; defaults to stdin')     parser.add_argument('-m
  -     def __new__(cls, value, **kwargs):         return valueclass ExtSlice(slice
  - line.endswith('\r\n') else '\n'         line = self.prev
  - [:12].lower().replace("_", "-")     if enc == "utf-8"

## Feature 26 — list, encoding, else
- mean: 0.6784
- max: 1.9641
- frequency: 0.9723
  - _flashed_messages=get_flashed_messages,             config=self.config,
  - rv, (dict, list)):                 rv = self.json.response(rv)             
  -  " + encoding             else:                 msg = "unknown encoding for {!r}: {}".
  -             if filename is not None:                 msg = '{} for {!r}'.format(
  -                  if filename is None:                     msg = 'encoding problem: utf-8'                 

## Feature 617 — import, token, which
- mean: 0.6818
- max: 3.5118
- frequency: 0.9672
  - import os import sys import typing as t import weakref from datetime import timedelta 
  - akes the same arguments as Werkzeug's         :class:`~werkzeug.
  -  lists of AST objects, or primitive ASDL types         # like identifiers and constants.         if
  - ples with these members: the token type; the     token string; a 2-tuple (s
  -  this shouldn't occur, since it now uses a         308 status which tells the browser to resend

## Feature 250 — return, false, elif
- mean: 0.6771
- max: 2.3553
- frequency: 0.9694
  - :                     if isinstance(item, AST):                         self.visit(item)             elif
  -         filename = None     bom_found = False     encoding = None     default =
  -                          elif not isinstance(value, AST):                             new_values.extend(value)
  - ], (Headers, dict, tuple, list)):                     rv, headers = rv  # pyright
  - indents = []         toks_append = self.tokens.append         startline =

## Feature 723 — _wrapper=self.ensure_sync, octnumber, exception=e
- mean: 0.6697
- max: 2.1893
- frequency: 0.9774
  - _wrapper=self.ensure_sync, exception=e)         propagate = self.config
  - ])*\.(?:[0-9](?:_?[0-9])*)?',
  - )*|[1-9](?:_?[0-9])*)' Intnumber =
  -  = group(r'[0-9](?:_?[0-9])*\.(
  - _wrapper=self.ensure_sync, response=response             )         except Exception: 

## Feature 297 — 9](?:_?[0-9])*, floatnumber, called
- mean: 0.6734
- max: 2.2862
- frequency: 0.9716
  - 9](?:_?[0-9])*' Pointfloat = group(r'[
  - 9](?:_?[0-9])*)' Intnumber = group(Hexnumber,
  - 9](?:_?[0-9])*[jJ]', Floatnumber + r'
  - 9](?:_?[0-9])*' + Exponent Floatnumber = group(
  -  t.Any] = req.view_args  # type: ignore[assignment]         

## Feature 8 — else, tokval, toknum
- mean: 0.6687
- max: 2.0658
- frequency: 0.9742
  - MIDDLE}:                 tokval = self.escape_brackets(tokval)#
  - :             _anchor = _url_quote(_anchor, safe="%!#$&'
  - able):             toknum, tokval = tok[:2]             if toknum ==
  - zeug.debug.preserve_context"](ctx)if (                 error is not None 
  - else:                 rv.status_code = status# extend existing headers with provided headers         if

## Feature 313 — none, url_adapter, string
- mean: 0.6703
- max: 2.0297
- frequency: 0.9715
  - .TokenizerIter(source, encoding=encoding, extra_tokens=extra_tokens)     
  -  " string. String = group(StringPrefix + r"'[^\n'\\]*(?:\\
  - (request.environ, request.trusted_hosts)  # pyright: ignore             
  -  mode='exec', *,           type_comments=False, feature_version=None, optimize=-
  -  1         col_offset = node.col_offset         end_col_offset = node

## Feature 641 — t.any, **kwargs, *args
- mean: 0.6720
- max: 2.4583
- frequency: 0.9679
  - def wrapper(self: Flask, *args: t.Any, **kwargs: t.Any
  -  test_cli_runner(self, **kwargs: t.Any) -> FlaskCliRunner:
  - def wrapper(self: Flask, *args: t.Any, **kwargs: t.Any
  - , *args: t.Any, **kwargs: t.Any) -> t.Any: 
  -  test_request_context(self, *args: t.Any, **kwargs: t.Any

## Feature 21 — none, elif, nargs=
- mean: 0.6683
- max: 2.3473
- frequency: 0.9731
  -      #: .. versionadded:: 0.8     session_interface: SessionInterface = SecureCookie
  - ', nargs='?', default='-',                         help='the file to parse; defaults to
  - indent=None, show_empty=False, ):     """     Return a formatted dump
  -         _external: bool | None = None,         **values: t.Any,     
  -                 server_name = None             elif not self.subdomain_matching:                 # W

## Feature 721 — return, false, else
- mean: 0.6668
- max: 1.9437
- frequency: 0.9747
  - should return one line of input as bytes.  Alternatively, readline     can be a callable
  -  `end_col_offset`) is missing, return None.If *padded* is `True
  - compare_fields(a, b):         return False     if compare_attributes and not _
  - , exc_info)         server_error: InternalServerError | ft.ResponseReturnValue 
  -  always be an ENCODING token     which tells you which encoding was used to decode the bytes stream

## Feature 755 — none, type, self.response_class.force_type
- mean: 0.6646
- max: 2.2585
- frequency: 0.9770
  - :                     rv = self.response_class.force_type(                         rv,  #
  -             else:                 url_adapter = self.create_url_adapter(None)if url
  - , list)):                 rv = self.json.response(rv)             elif isinstance(rv
  -  response.         """         ctx = self.request_context(environ)         error
  -                     rv = self.response_class.force_type(                         rv,  # type:

## Feature 44 — error, none, return
- mean: 0.6648
- max: 2.2082
- frequency: 0.9750
  - (before_func)()if rv is not None:                         return rv  # type: ignore
  - .request.blueprints)if handler is not None:             server_error = self.ensure
  -  Set the name of the Click group in case someone wants to add         # the app's commands to
  -  default this will invoke the         registered error handlers and fall back to returning the         exception as response
  - with('\r\n') else '\n'         line = self.prev_line.

## Feature 52 — called, value, this
- mean: 0.6605
- max: 2.1279
- frequency: 0.9786
  -  events may not be             called depending on when an error occurs during dispatch.:param environ:
  - J]', Floatnumber + r'[jJ]') Number = group(Imagnumber
  - .lineno         if 'end_lineno' in node._attributes:             if get
  - Value:         """This method is called whenever an exception occurs that         should be handled. A
  - cast(Response, rv)         # prefer the status if it was provided         if status is

## Feature 626 — return, type, ignore[no-any-return]def
- mean: 0.6623
- max: 2.1384
- frequency: 0.9751
  - return None         encoding = _get_normal_name(match.group(1).decode())
  - )     return outdef _get_normal_name(orig_enc):     """I
  - , str):         text = node.value     else:         return None     if
  - debug or self.config["TRAP_BAD_REQUEST_ERRORS"]         ):             
  - , list)):                 rv = self.json.response(rv)             elif isinstance(rv

