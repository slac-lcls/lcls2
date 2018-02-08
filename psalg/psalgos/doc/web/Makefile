# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
MYPKGNAME     = psalgos
SPHINXOPTS    =
SPHINXBUILD   = python -msphinx
SPHINXPROJ    = $(MYPKGNAME)-doc
SOURCEDIR     = .
#BUILDDIR      = _build
BUILDDIR      = ../../../$(MYPKGNAME)-doc

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

helpme:
	@echo "  helpme      prints this help info"
	@echo "  clean       to clean build doc directory $(BUILDDIR)"
	@echo "  docdir      to create documentation directory $(BUILDDIR), clone repo"
	@echo "  html        to generate html documentation"
	@echo "  gh-pages    to add, commit and push documentation to gh-pages"
	@echo "  newdoc      sequence of commands: helpme, clean, docdir, html, gh-pages"
	@echo 

printpars:
	@echo
	@echo "MAKE      : $(MAKE)"
	@echo "MYPKGNAME : $(MYPKGNAME)"
	@echo "SPHINXPROJ: $(SPHINXPROJ)"
	@echo "BUILDDIR  : $(BUILDDIR)"
	@echo "Sequence of make commands: clean, docdir, html, gh-pages, update"
	@echo

clean:
	rm -rf $(BUILDDIR)
	@echo "Documentation directory $(BUILDDIR) is removed"

update:
	cd ../.. && ls src && rm src/*.pyc && git pull
	@echo "Repo is updated from github"

docdir:
	mkdir $(BUILDDIR) && cd $(BUILDDIR) && rm -rf * && git clone https://github.com/lcls-psana/$(MYPKGNAME).git html && cd html && git branch gh-pages && git symbolic-ref HEAD refs/heads/gh-pages && rm .git/index && git clean -fdx
	@echo
	@echo "Empty documentation directory is ready: $(BUILDDIR)"

gh-pages:
	cd $(BUILDDIR)/html && git add -A && git commit -m "add/update doc" && git push origin gh-pages -f
	@echo
	@echo "Documentation is pushed on gh-pages"

newdoc:
	$(MAKE) helpme && $(MAKE) clean && $(MAKE) docdir && $(MAKE) html && $(MAKE) gh-pages && $(MAKE) -i -k update
	@echo
	@echo "Documentation is built and committed, repo is updated."

