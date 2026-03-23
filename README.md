# MGFD40

## Do Ctrl+Shift+V on this README.md to view preview

## Installation

Choose **Option A** (GitHub Desktop — recommended for beginners) or **Option B** (command line) for Installation and Contributing.

---

### Option A: GitHub Desktop (Recommended)

#### A1. Install GitHub Desktop
Download and install from [desktop.github.com](https://desktop.github.com/).

#### A2. Sign in to GitHub
Open GitHub Desktop → **File > Options > Accounts** → sign in with your GitHub account.

#### A3. Clone the repository
1. In GitHub Desktop, click **File > Clone repository**
2. Select the **URL** tab and paste: `https://github.com/hillaryTse/MGFD40-Final-Project.git`
3. Choose a local path and click **Clone**

#### A4. Install Python
Download **Python 3.10 or newer** from [python.org/downloads](https://www.python.org/downloads/).
On Windows, check **"Add Python to PATH"** during installation.

#### A5. Install required libraries
Open a terminal (or the terminal inside VS Code) in the project folder and run:
```bash
pip install pandas pyarrow praw requests
```

---

### Option B: Command Line

#### B1. Install Git
Download from [git-scm.com](https://git-scm.com/downloads) and install.

#### B2. Clone the repository
```bash
git clone https://github.com/hillaryTse/MGFD40-Final-Project.git
cd MGFD40
```

#### B3. Install Python
Download **Python 3.10 or newer** from [python.org/downloads](https://www.python.org/downloads/).
On Windows, check **"Add Python to PATH"** during installation.

#### B4. Install required libraries
```bash
pip install pandas pyarrow praw requests
```

---

## Contributing

### Option A: GitHub Desktop

1. **Pull the latest changes** before starting work:
   - In GitHub Desktop, click **Fetch origin** (top bar), then **Pull origin** if there are new changes

2. **Create a branch** (recommended if more than one person is working, or if you want someone to review before merging):
   - In GitHub Desktop, click **Current Branch** at the top → **New Branch**
   - Name it `your-name/feature-description` and click **Create Branch**

2. **Make your changes** in VSCode.

3. **Commit your changes**:
   - In GitHub Desktop, review changed files in the left panel
   - Write a short summary in the **Summary** box (bottom-left)
   - Click **Commit to your-name/feature-description**

4. **Push your branch**:
   - Click **Push origin** (top bar)

5. **Open a Pull Request** (only if you created a branch):
   - Click **Create Pull Request** — this opens GitHub in your browser
   - Add a description and submit for team review

> If you did **not** create a branch, just commit and click **Push origin** — no Pull Request needed.

---

### Option B: Command Line

1. Pull the latest changes before starting work:
```bash
git pull origin
```

2. Create a branch for your changes -- (**do this step if you want someone to review before saving to repository OR if more than 1 person is working on code at the same time**):
```bash
git checkout -b your-name/feature-description
```

3. Make your changes, stage and commit:
```bash
git add file.py file.csv file.parquet folder/
git commit -m "Brief description of what you did"
```

4. If you created a branch previously, push your branch:
```bash
git push origin your-name/feature-description
```
but if you did not create a branch:
```bash
git push origin
```

5. Open a **Pull Request** on GitHub for the team to review before merging -- (**Not required if you did not create a branch previously**)

---

### Guidelines
- Write clear commit messages explaining what and why
