
Developers' Documentation
=========================

Running Tests
^^^^^^^^^^^^^

Running tests requires the ``pytest`` package, available via ``pip`` or ``conda``.

Go into the MemoCell directory and run::

   # run all fast tests
   python setup.py pytest

Alternatively, go into the ``./tests/`` directory and run::

   # run all fast tests
   pytest

To run test scripts individually (within the ``./tests/`` directory)::

   # adapt line to run an individual script
   pytest test_memocell_<some tests>.py

Some tests are marked as slow as they are expensive; these are typically skipped
during testing. To include them, add the runslow option (within the ``./tests/`` directory)::

   # run all tests (fast and slow)
   pytest --runslow

   # all tests (fast and slow) of an individual script
   pytest test_memocell_<some tests>.py --runslow

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

Requirements are ``sphinx``, ``numpydoc`` and the ``sphinx_rtd_theme``, available via ``pip``. Go into the ``./docs/`` directory and run::

   make html

A local html file can then be found at ``./docs/_build/html/index.html``.


Performance Notes
^^^^^^^^^^^^^^^^^

If MemoCell appears slow, there may be two major reasons for that (maybe more that we are
currently not aware of).

First, MemoCell relies heavily on efficient ODE integration by ``odeint``,
where the moment equations are computed based on ``numba`` just-in-time compilation. So make sure
that the moment equations are actually computed in the compiled form, which can be a ~100-fold difference in performance.

Second, when running larger model selections with the parallel option, it seems critical for
performance that the right number of parallel processes is used for the respective system. MemoCell is typically
CPU-heavy and has little Input/Output tasks, hence typically one should not use more parallel processes than
physical cores of the system (current default). However to know better, one might be advised to do test runs
for different process numbers. Less can be more.

Note, that the tqdm progress bar often updates in larger chunks in parallel mode, particularly when
many parallel processes are used. This can be a bit misleading, but it is not relevant for the
actual performance.


Release Workflow
^^^^^^^^^^^^^^^^

Increment the version number in ``setup.py`` file to the version that shall be
released now. Also adapt the version in ``memocell/__init__.py`` and
``docs/conf.py``. Commit/push to GitHub.

Go into the local MemoCell package folder and remove old egg files via
``rm -rf build *.egg-info``. Also remove the previous distributions by
removing the ``dist`` folder via finder or terminal.

Then build the distribution/wheels via ``python setup.py sdist bdist_wheel``.
Check with twine if it has run correctly via ``twine check dist/*``.

Upload the release to pypi `test` registry via
``twine upload --repository-url https://test.pypi.org/legacy/ dist/*``.

If everything looks good, make a release on GitHub; tag versions starting
with ``v<...>``. Upload the two files (source dist and wheels) from the
``dist`` folder in that process.

Finally, use ``twine upload dist/*`` to upload to the actual pypi registry.

Travis should automatically trigger a build test after these processes.
Check if it runs through and that the pypi and travis badges update
accordingly on the GitHub homepage.
