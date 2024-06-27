Installation guide
==================

1. Install `Anaconda <https://www.anaconda.com/download>`_ or `Miniconda <https://docs.conda.io/projects/miniconda/en/latest/index.html#latest-miniconda-installer-links>`_ 
    Anaconda is the standard **package manager** for Python in the scientific 
    community. It comes with a GUI for user-friendly package installation 
    and management. However, here we describe its use through the terminal. 
    Miniconda is a lightweight implementation of Anaconda without the GUI.

2. Open a **terminal**
    Roughly speaking, a terminal is a **text-based way to run instructions**. 
    On Windows, use the **Anaconda prompt**, you can find it by searching for it. 
    On macOS or Linux you can use the default Terminal app.

3. **Clone the source code** with the following command:
   
    .. code-block:: 
    
        git clone https://github.com/ElpadoCan/ChromRings.git

    .. important::
    
        On Windows, if you get the error ``ERROR: Cannot find the command 'git'`` 
        you need to install ``git`` first. Close the terminal and install it 
        from `here <https://git-scm.com/download/win>`_. After installation, 
        you can restart from here, but **remember to activate the** ``acdc`` 
        **environment first** with the command ``conda activate acdc``.

4. **Navigate to the ChromRings folder** with the following command:
   
    .. code-block:: 
   
        cd ChromRings

    The command ``cd`` stands for "change directory" and it allows you to move 
    between directories in the terminal. 

5. **Update conda** with the following command:
   
    .. code-block:: 

        conda update conda
    
    This will update all packages that are part of conda.

6. Create a **virtual environment** with the following command:
   
    .. code-block:: 
    
        conda create -n acdc python=3.10

    This will create a virtual environment, which is an **isolated folder** 
    where the required libraries will be installed. 
    The virtual environment is called ``acdc`` in this case.

7. **Activate the virtual environment** with the following command:
   
    .. code-block:: 
    
        conda activate acdc

    This will activate the environment and the terminal will know where to 
    install packages. 
    If the activation of the environment was successful, this should be 
    indicated to the left of the active path (you should see ``(acdc)`` 
    before the path).

    .. important:: 

       Before moving to the next steps make sure that you always activate 
       the ``acdc`` environment. If you close the terminal and reopen it, 
       always run the command ``conda activate acdc`` before installing any 
       package. To know whether the right environment is active, the line 
       on the terminal where you type commands should start with the text 
       ``(acdc)``.

8. **Update pip** with the following command:
   
    .. code-block:: 
   
        python -m pip install --upgrade pip
    
    While we could use conda to install packages, Cell-ACDC is not available 
    on conda yet, hence we will use ``pip``. 
    Pip the default package manager for Python. Here we are updating pip itself.

9.  **Install ChromRings** with the following command:
   
    .. code-block:: 
   
        pip install -e "."

    The ``.`` at the end of the command means that you want to install from 
    the current folder in the term
