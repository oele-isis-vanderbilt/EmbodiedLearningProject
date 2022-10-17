# EmbodiedLearningProject
Official Repository for Embodied Learning Project

## :telescope: Data

The ``data`` folder is not included in the GitHub repository for memory
size limitations. We are using [DVC](https://dvc.org/) to
more easily synchronize data between team members. The data is self-hosted
in LIVE server 1 (10.28.128.86), where you can login with your VUID
credentials. Here are the instructions to acquire the data.

First, install DVC with version 2.5.4 with the command below. This version
was selected as a safety precaution for DVC's latest release is having some
issues with SSH self-hosted options.

```
pip install dvc==2.5.4
```

Once dvc is installed, add your VUID credentials with the following command:

```
dvc remote modify --local live1-server user {your VUID username}
dvc remote modify --local live1-server password {your VUID password}
```

> :warning: Make sure to use `--local` in the command above to storing your
password locally and avoid storing it within the GitHub repo

Now you have finish configuring DVC, you should be able to execute dvc commands
(e.g. ``add``, ``commit``, ``push``, and ``pull``) to perform version control
operations on the data. To start, use the following command to obtain
the latest version of the data:

```
dvc pull
```
