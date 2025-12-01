# General Unit Test Instructions

When writing unit tests for the `sigalg` library, please adhere to the following guidelines to ensure consistency and maintainability across the codebase:

## Imports

* Place `import sigalg as sa` at the top of each unit test file.
* Import `pytest` and use it as the testing framework. Use fixtures where appropriate.

## Assertions

* Use `assert` statements for all validations.
* If an object has a `values` property, that is an object from pandas like a DataFrame, Series, or Index. Please use the methods in `pd.testing` to validate those values. For example, make a data frame called `expected_df` and use `pd.testing.assert_frame_equal(object.values, expected_df)` to validate.

## Formatting and Naming Conventions

* Do not put any comments or docstrings in your unit tests.
* **Always include exactly one blank line between method definitions within a class.** The only place you should NOT have blank lines is inside the body of a function or method.
* **Always include exactly one blank line between fixture definitions and the first test method in a class.**
* Always include **two** blank lines between class definitions (this is standard Python formatting).
* Name unit test classes starting with the word "Test" followed by a description of what is being tested, using Pascal case (e.g. `TestSampleSpaceInitialization`).
* Name unit test methods starting with the word "test" followed by a description of what is being tested, using snake case (e.g. `test_sample_space_initialization`).
* Use `fss` and `fps` as variable names for `FeaturizedSampleSpace` and `FeaturizedParameterSpace` objects respectively in unit tests.
* Use `probabilities` as the variable name for probability dictionaries in unit tests.
* Use `prob_measure` as the variable name for `ProbabilityMeasure` objects in unit tests.
* Use `sigma_algebra` as the variable name for `SigmaAlgebra` objects in unit tests.
* Use `sample_space` as the variable name for `SampleSpace` objects in unit tests.
* Use `prob_space` as the variable name for `ProbabilitySpace` objects in unit tests.

## Specific Class Organization

* Make a class called `TestConstructor` for all constructor-related unit tests.
* Place all unit tests for properties in a separate class called `TestProperties`.
* Place all validation unit tests in a separate class called `TestValidation`.
* Place all "setter method" unit tests in a separate class called `TestSetters`.
* Make sure to write unit tests for equality methods (`__eq__`), if available. Place these tests in a separate class called `TestEquality`. If no equality method exists, and you think it should, notify me and write the method as well as the unit tests.

## Exclusions

* Do not write unit test for `__repr__` methods.