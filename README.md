# counting-cockroaches
Using social media to assess the severity of service failures.

## Getting Started
* Unlimited private repositories are available for students and academics: https://education.github.com.
* GitHub Guides is a good place to get familiar with how to use the platform: https://guides.github.com.

## Code Collaboration for R/RStudio
The [GitHub Flow](https://guides.github.com/introduction/flow/) is straightforward enough, but how it integrates with R and RStudio is something I'm still figuring out. I'll provide more guidance at a future date, but here's what I'm using to walk through this process:

* Hadley Wickham's *R Packages* book: http://r-pkgs.had.co.nz/git.html.
* Happy Git and GitHub for the useR: http://happygitwithr.com.

# Connecting RStudio to GitHub
## Prerequisites
* Install/update R & RStudio
* Install Git (see next section)

## Installing Git
### Windows
* Download Git for Windows at https://git-for-windows.github.io/
* You can accept all the default settings during installation.
* Note: RStudio for windows likes for git to be in the Files(x86) folder. If not in this location, RStudio may not detect it, and may cause headaches for you later.
### Mac OS
* Download Git for Mac OS at http://git-scm.com/downloads

## Create SSH key pair
* this is so the GitHub repo will trust changes made by your computer
* First check that running "file.exists("~/.ssh/id_rsa.pub")" in R returns FALSE. If so, continue.
* Tell Git your name and email address (the same one as your GitHub account) by typing each of the following lines in the terminal:

git config --global user.name "YOUR FULL NAME"

git config --global user.email "YOUR EMAIL ADDRESS"

* Next go to RStudio preferences, choose the Git/SVN panel, and click “Create RSA key…”:
* You can go back and click the "View Public Key" button to get see your public key
* Give GitHub your SSH public key: https://github.com/settings/ssh.

## Create Repo
* Create a new project (dropdown menu found in upper-right hand corner of RStudio)
* Select “Version Control”, then “Git”
* Copy and paste the URL of the GitHub repository
* Enter "counting-cockroaches" as the name and choose the directory (the folder where you are storing files for your RA work)
* Click "Create Project" (you may be prompted to enter your GitHub username and password)
* You should now have a "Git" pane next to your "Environment" and "History"

## Now What?
* You'll notice three actions in the Git pane in RStudio : Commit, Pull, and Push
* Your first step should always be Pull, which brings your local branch up to date with the GitHub master
* After making changes to a file, you need to Commit it (by checking the box of the filename in the Git pane and clicking "Commit") and add a short, informative description.
* After Commiting, you will see a little message in the Git pane that your branch is ahead of the master branch by one or more commits
* When you see this message, you need to update the master branch by Pushing your changes (click on the Push button)
