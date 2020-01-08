Installation
=============

Prerequisites
-------------

- Install Anaconda_
- Install git_

Linux
------

- Download the `SCOUT repository`_ directly from Github or using git:

.. code-block:: bash

    git clone https://github.com/chunglabmit/scout.git

- Install the *scout* conda environment:

.. code-block:: bash

    cd scout/
    conda env create -f environment.yml


- Activate the *scout* environment and install SCOUT in editable mode:

.. code-block:: bash

    conda activate scout
    pip install -r requirements.txt
    pip install . -e

Windows
--------
**Note**: Windows users need to use the "Anaconda Prompt" terminal if *conda* is not available from your PATH variable.

- Download the `SCOUT repository`_ directly from Github or using git:

.. code-block:: bash

    git clone https://github.com/chunglabmit/scout.git

- Create a *scout* conda environment:

.. code-block:: bash

    conda create -n scout python=3.6

- Install required dependencies:

.. code-block:: bash

    cd scout/
    conda activate scout
    pip install -r requirements.txt

**Note**: Some dependencies require a bit more work to install on Windows. We may include a Dockerfile in the future.

MacOS
------
SCOUT has not been tested with MacOS, but the Linux install instructions may work.

.. _Anaconda: https://www.anaconda.com/
.. _git: https://git-scm.com/downloads
.. _SCOUT repository: https://github.com/chunglabmit/scout

Updates
--------

Installing SCOUT in editable mode makes updating easy. You can simply pull the latest version from Github:

.. code-block:: bash

    git pull  # Run inside scout installation folder
