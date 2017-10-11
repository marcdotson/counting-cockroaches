# counting-cockroaches
Using social media to assess the severity of service failures.

## Getting Started
* Unlimited private repositories are available for students and academics: https://education.github.com.
* GitHub Guides is a good place to get familiar with how to use the platform: https://guides.github.com.
* Make sure you have the latest version of R, RStudio, and Git (see next section) installed.

## Installing Git
### Windows
* Download Git for Windows at https://git-for-windows.github.io/.
* Accept all default settings during installation.
* Note that RStudio for Windows likes for Git to be in the Files(x86) folder. If it is not in this location, RStudio may not detect it, which may cause problems later.

### Mac OS
* Download Git for Mac OS at http://git-scm.com/downloads.

## Obtaining SSH RSA Key
* The SSH RSA Key is how GitHub authenticates commits from your computer. First, see if you have a key generated by running the following code in R:

> file.exists("~/.ssh/id_rsa.pub")

### Code Returns TRUE
* If the code returns TRUE, then you already have a key that you can use.
* Go to RStudio preferences, choose the Git/SVN panel, and click "View Public Key."
* Give GitHub your SSH key: https://github.com/settings/ssh.

### Code Returns FALSE
* If the code returns FALSE, then you need to generate a key.
* Tell Git your name and email address (the same one as your GitHub account) by typing each of the following lines in the terminal:

> git config --global user.name "YOUR FULL NAME"
> git config --global user.email "YOUR EMAIL ADDRESS"

* Go to RStudio preferences, choose the Git/SVN panel, and click “Create RSA Key.”
* Give GitHub your SSH key: https://github.com/settings/ssh.

## Creating a Local Branch
* Create a new project (drop-down menu found in upper-right hand corner of RStudio).
* Select “Version Control”, then “Git.”
* Copy and paste the URL of the GitHub repository.
* Enter the name of your local branch (e.g., "counting-cockroaches") and choose the directory where you want it saved.
* Click "Create Project" (you may be prompted to enter your GitHub username and password).
* You should now have a "Git" pane next to your "Environment" and "History" panes in RStudio.

## Now What?
* You have three important buttons in the Git pane in RStudio: Commit, Pull, and Push.
* Always statrt with a Pull, which brings your local branch up to date with the master branch.
* After making changes to a file, you need to Commit it (by checking the box of the filename in the Git pane and clicking "Commit") and add a short, informative description.
* After commiting, you will see a little message in the Git pane that your branch is ahead of the master branch by one or more commits.
* When you see this message, you need to submit an update to the master branch by Pushing your changes (click on the Push button).

## References
* Hadley Wickham's *R Packages* book: http://r-pkgs.had.co.nz/git.html.
* Happy Git and GitHub for the useR: http://happygitwithr.com.
