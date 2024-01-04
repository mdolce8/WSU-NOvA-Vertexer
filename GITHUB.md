# Working with GitHub

### How to work with `git` (and GitHub) 


---

To commit to this repository, the process is:

### 1. Create a new git branch for your work ( # assuming you are already on `main` branch)

``` git branch -b checkout <branch_name>```

For consistency with NOvA, use the following naming convention for your branch name:

``` feature/your_name>_<brief_description> ```


### 2. Make your changes and commit them to your branch

``` git add <files> ```

``` git commit -m "commit message" ```


### 3. Push your branch to the remote repository

``` git push origin <branch_name> ```


### 4. Create a pull request on GitHub to merge your branch into `main`.

Allow code to be reviewed and make any necessary changes suggested by the reviewers.

After that, all done. Code is merged into `main` and your branch can be deleted.

``` git branch -d <branch_name> ```

