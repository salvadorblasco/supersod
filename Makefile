PYUIC=pyuic5
SRCDIR=src
UIDIR=widgets
TESTDIR=tests
PY=python3

all: deps gui

deps:
	pip3 install -r requirements.txt

doc:
	cd doc
	make html

gui:
	$(PYUIC) $(UIDIR)/mainwindow.ui   > $(SRCDIR)/ui_mainwindow.py
	$(PYUIC) $(UIDIR)/points.ui       > $(SRCDIR)/ui_points.py
	$(PYUIC) $(UIDIR)/importdialog.ui > $(SRCDIR)/ui_import.py

test:
	cd $(TESTDIR)
	$(PY) test.py
	$(PY) testconsol.py

.PHONY: deps test
