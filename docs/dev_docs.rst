
Developers' Documentation
=========================

Running Tests
^^^^^^^^^^^^^

Running tests requires the ``pytest`` package, available via ``pip`` or ``conda``.

Go into the Memopy directory and run::

   # run all fast tests
   python setup.py pytest

Alternatively, go into the ``./tests/`` directory and run::

   # run all fast tests
   pytest

To run test scripts individually (within the ``./tests/`` directory)::

   # adapt line to run an individual script
   pytest test_memopy_<some tests>.py

Some tests are marked as slow as they are expensive; these are typically skipped
during testing. To include them, add the runslow option (within the ``./tests/`` directory)::

   # run all tests (fast and slow)
   pytest --runslow

   # all tests (fast and slow) of an individual script
   pytest test_memopy_<some tests>.py --runslow

Test Coverage
^^^^^^^^^^^^^

A test coverage metric can be obtained by the ``coverage`` package, available
via ``pip install coverage``.

Then run desired tests under coverage, for example (within the ``./tests/`` directory)::

   # run all tests under coverage
   coverage run -m pytest --runslow

To get a coverage report, run one of the following options::

   # coverage report in the terminal
   coverage report -m

   # create folder with html report (open index.html)
   coverage html


Building Documentation
^^^^^^^^^^^^^^^^^^^^^^

Requirements are ``sphinx`` and the ``sphinx_rtd_theme``, available via ``pip``. Go into the ``./docs/`` directory and run::

   make html

A local html file can then be found at ``./docs/_build/html/index.html``.
