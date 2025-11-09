# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a **Japanese-language tutorial repository** for the statsmodels Python library. It contains 47 markdown tutorial files organized into 12 chapters covering statistical modeling, hypothesis testing, and data analysis.

**Target Audience**: Beginners to intermediate users learning statsmodels
**Language**: Japanese (all documentation and code comments)
**License**: MIT

## Repository Structure

The repository follows a chapter-based organization:

```
XX_topic_name/
├── 01_subtopic.md
├── 02_subtopic.md
└── ...
```

- **Chapters 1-2**: Comprehensive tutorials with detailed explanations, sample code, expected outputs, and practice problems with solutions
- **Chapters 3-11**: Practical code examples demonstrating key statsmodels features
- **Chapter 12**: Real-world case studies (economics, healthcare, marketing, finance)

### Content Depth by Chapter

- **Detailed** (第1-2章): Introduction and Linear Regression - extensive beginner-friendly content
- **Practical** (第3-11章): Intermediate topics with working code samples
- **Applied** (第12章): Case studies with complete analytical workflows

## Tutorial Content Guidelines

### When Adding or Modifying Tutorials

1. **Code Format**:
   - Include detailed Japanese comments for beginners
   - Show complete, runnable examples
   - Provide expected output examples in code blocks
   - Use consistent import patterns:
     ```python
     import numpy as np
     import pandas as pd
     import statsmodels.api as sm
     import statsmodels.formula.api as smf
     ```

2. **Structure for Detailed Chapters (1-2)**:
   - Introduction with theory
   - Sample code with line-by-line comments
   - Output examples
   - Interpretation of results
   - Practice problems (練習問題)
   - Model solutions (模範解答)

3. **Structure for Practical Chapters (3-11)**:
   - Brief explanation
   - Working code example
   - Key function/method usage

4. **Case Studies (Chapter 12)**:
   - Complete data generation
   - Full analytical workflow
   - Visualization
   - Statistical testing
   - Interpretation and conclusions

### Naming Conventions

- Files: `##_topic_name.md` (e.g., `01_simple_regression.md`)
- Directories: `##_category/` (e.g., `02_linear_regression/`)
- Consistent numbering across all chapters

## Git Workflow

### Commit Messages

Use conventional commit format in Japanese when appropriate:

```
feat: 新しいチュートリアルを追加
docs: README更新
chore: ライセンスファイル追加
```

### Version Updates

When adding significant content, update:
1. `README.md` - 更新履歴セクション
2. Version number in changelog
3. Commit with descriptive message

## Content Quality Standards

### For Tutorial Files

- **Completeness**: All code must be executable
- **Clarity**: Explain statistical concepts in Japanese for beginners
- **Accuracy**: Verify output examples match code
- **Pedagogy**: Include "why" not just "how"

### For Practice Problems

- Provide realistic scenarios
- Include clear instructions
- Always provide complete solutions
- Solutions should demonstrate best practices

## Statistical Content Notes

### Model Assumptions

When writing about statistical models, always mention:
- Model assumptions (線形性、等分散性、正規性 etc.)
- Diagnostic tests
- Interpretation guidelines
- When to use vs. not use the method

### Common statsmodels Patterns

- **Formula API**: `smf.ols('y ~ x1 + x2', data=df).fit()`
- **Array API**: `sm.OLS(y, X).fit()` (requires manual constant addition)
- **Diagnostics**: Use `.summary()`, `.resid`, `.fittedvalues`
- **Testing**: Include proper statistical tests (Shapiro-Wilk, Breusch-Pagan, etc.)

## Learning Path Reference

When suggesting content improvements or additions, consider the learning paths:

- **Beginners**: Ch 1 → 2 → 3 → 5
- **Intermediate**: Ch 4 → 6 → 7 → 12
- **Advanced**: Ch 8 → 9 → 10 → 11

New content should fit appropriately into these progressions.

## Dependencies

Required packages (as documented in README.md):
```bash
pip install statsmodels pandas numpy matplotlib seaborn scipy jupyter
```

All code examples should work with these dependencies without additional installations.

## Repository Maintenance

### When Reviewing Changes

- Verify all code is executable
- Check Japanese grammar and clarity
- Ensure consistent formatting
- Validate statistical accuracy
- Test that output examples are realistic

### File Updates

- Keep README.md table of contents synchronized
- Update version history for significant changes
- Maintain consistent markdown formatting
- Preserve the beginner-friendly tone in detailed chapters
