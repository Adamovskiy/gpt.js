---
description: 
alwaysApply: true
---

# Code Style and Development Guidelines for Claude AI Assistant

This document outlines the coding standards and best practices for working with this TypeScript/React project.

## Language Requirements

- **English Only**: All code, comments, documentation, commit messages, variable names, function names, and any text within the repository must be in English.
- No exceptions for mixed languages in code or documentation.

## Type Safety

### Avoid `any` and Type Assertions
- **Never use `any`** unless there is an extremely compelling reason with detailed justification in comments
- **Avoid type assertions** (`as Type`) when possible - prefer type guards and proper typing
- Use strict TypeScript configuration and embrace type checking
- Prefer union types and discriminated unions over loose typing

### Proper Generics Usage
- Use generics to maintain type safety instead of `any`
- Parameterize interfaces and classes with appropriate generic constraints
- Example: `interface Tokenizer<T>` instead of `interface Tokenizer { getSerializedData(): any }`

## Error Handling

### Linter and Type Errors
- **Do not ignore linter errors** without addressing the root cause
- **Do not suppress TypeScript errors** with `@ts-ignore` or similar unless absolutely necessary
- Always run `npm run lint` to check for linting issues before considering the work complete
- Fix type errors properly rather than working around them

### Testing and Validation
- Use `npm run lint` as the primary tool for checking code quality
- Address all linting errors and warnings before submitting code
- Ensure TypeScript compilation passes without errors

## Development Workflow

### Local Development
- **Do not attempt to start the dev server** unless explicitly requested by the user
- Focus on code correctness and type safety over immediate testing
- Use linting and type checking tools instead of runtime testing for validation

## Component Architecture

### React Components
- **Keep components under ~200 lines** of code
- Break down large components into smaller, focused sub-components
- Separate concerns: logic, presentation, and state management
- Use meaningful component names that describe their purpose

### File Organization
- Group related functionality together
- Use clear, descriptive file and directory names
- Maintain consistent naming conventions throughout the project

## Code Quality Standards

### Readability
- Write self-documenting code with clear variable and function names
- Add comments only when the code's intent is not obvious
- Use consistent formatting and indentation
- Follow established patterns within the codebase

### Performance
- Consider performance implications of architectural decisions
- Use appropriate data structures and algorithms
- Minimize unnecessary re-renders and computations
- Leverage TypeScript's compile-time optimizations

## Best Practices Summary

1. **English only** - no mixed languages anywhere in the repository
2. **Strict typing** - avoid `any` and type assertions
3. **Fix errors properly** - don't ignore linter or type errors
4. **Use `npm run lint`** for code quality checks
5. **Don't start dev server** unless explicitly requested
6. **Keep components small** - under 200 lines
7. **Embrace TypeScript** - use its full type safety features
8. **Write clean code** - readable, maintainable, and well-structured

Following these guidelines ensures code quality, maintainability, and consistency across the project.
