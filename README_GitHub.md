# How to Push this Project to GitHub

Follow these steps to upload this project to a GitHub repository on the `main` branch.

### 1. Create a Repository on GitHub
Go to [GitHub.com](https://github.com/new) and create a new empty repository. Do **not** initialize it with a README, .gitignore, or License (since you already have these).

### 2. Initialize Git (if not already done)
Open your terminal in the project root directory (`c:\Users\User\OneDrive - Alpen-Adria Universität Klagenfurt\WS26\Bachelor's Thesis`) and run:

```bash
git init
```

### 3. Add Files
Stage all your files for the first commit:

```bash
git add .
```

### 4. Commit Changes
Commit the files with a message:

```bash
git commit -m "Initial commit of Thesis Summarization App"
```

### 5. Rename Branch to Main
Ensure your default branch is named `main`:

```bash
git branch -M main
```

### 6. Connect to GitHub
Link your local repository to the remote GitHub repository. Replace `<YOUR_REPOSITORY_URL>` with the URL you copied from step 1 (e.g., `https://github.com/username/repo-name.git`).

```bash
git remote add origin <YOUR_REPOSITORY_URL>
```
*(If you get an error saying `remote origin already exists`, use `git remote set-url origin <YOUR_REPOSITORY_URL>` instead)*

### 7. Push to GitHub
Upload your code to the `main` branch:

```bash
git push -u origin main
```
