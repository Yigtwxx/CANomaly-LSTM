# Contributing to CANomaly-LSTM

First off, thank you for considering contributing to CANomaly-LSTM! It's people like you that make open source tools such a great place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

Following these guidelines helps to communicate that you respect the time of the developers managing and developing this open source project. In return, they should reciprocate that respect in addressing your issue or assessing your improvements.

## 📜 Table of Contents
- [Code of Conduct](#-code-of-conduct)
- [How Can I Contribute?](#-how-can-i-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Pull Request Process](#-pull-request-process)
- [Styleguides](#-styleguides)
  - [Python Styleguide](#python-styleguide)
  - [Commit Messages](#commit-messages)

---

## 🤝 Code of Conduct

This project and everyone participating in it is governed by the [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainer.

*(Note: If a CODE_OF_CONDUCT.md does not yet exist, we follow the standard [Contributor Covenant](https://www.contributor-covenant.org/).)*

---

## 🚀 How Can I Contribute?

### Reporting Bugs

This section guides you through submitting a bug report. Following these guidelines helps maintainers and the community understand your report, reproduce the behavior, and find related reports.

**Before Submitting a Bug Report:**
*   Check the [Issues](https://github.com/Yigtwxx/CANomaly-LSTM/issues) to see if the problem has already been reported.
*   Make sure your software is up to date.

**How to Submit a Good Bug Report:**
*   **Use a clear and descriptive title** for the issue to identify the problem.
*   **Describe the exact steps to reproduce the problem** in as many details as possible.
*   **Provide specific examples** to demonstrate the steps. Include copy/pasteable snippets, which you use in those examples.
*   **Describe the behavior you observed** after following the steps and point out what exactly is the problem with that behavior.
*   **Explain which behavior you expected to see instead** and why.
*   **Include screenshots or animated GIFs** which show you following the reproduction steps.

### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion, including completely new features and minor improvements to existing functionality.

**How to Submit a Good Enhancement Suggestion:**
*   **Use a clear and descriptive title** for the issue to identify the suggestion.
*   **Provide a step-by-step description of the suggested enhancement** in as many details as possible.
*   **Explain why this enhancement would be useful** to most users.

### 📥 Pull Request Process

1.  **Fork the repository** and create your branch from `main`.
2.  **Clone your fork** locally:
    ```bash
    git clone https://github.com/YOUR-USERNAME/CANomaly-LSTM.git
    ```
3.  **Create a branch** for your feature or bugfix:
    ```bash
    git checkout -b feature/amazing-feature
    # or
    git checkout -b fix/critical-bug
    ```
4.  **Make your changes**. Ensure you test them locally using the provided scripts (e.g., `src/generate_can_dataset.py` to ensure data generation still works).
5.  **Commit your changes** using descriptive commit messages (see Styleguides below).
6.  **Push to your fork**:
    ```bash
    git push origin feature/amazing-feature
    ```
7.  **Open a Pull Request**. Provide a clear description of the changes and link to any relevant issues.

---

## 🎨 Styleguides

### Python Styleguide

*   We follow **[PEP 8](https://www.python.org/dev/peps/pep-0008/)** guidelines.
*   Keep code clean and readable.
*   Use meaningful variable and function names.
*   Add comments for complex logic, but prefer self-documenting code.
*   **Type Hinting**: We encourage the use of Python type hints for function arguments and return values.

### Commit Messages

*   **Use the imperative mood** in the subject line (e.g., "Add feature" not "Added feature").
*   **Limit the subject line to 50 characters**.
*   **Capitalize the subject line**.
*   **Do not end the subject line with a period**.
*   Separate subject from body with a blank line.
*   Use the body to explain **what** and **why** vs. **how**.

## 📞 Questions?

Feel free to open an issue or reach out if you have any questions!

Happy coding! 🚗💻
