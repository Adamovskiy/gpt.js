import js from '@eslint/js';
import eslintConfigPrettier from 'eslint-config-prettier';
import tailwindcssPlugin from 'eslint-plugin-better-tailwindcss';
import importPlugin from 'eslint-plugin-import';
import { configs as perfectionistConfigs } from 'eslint-plugin-perfectionist';
import prettierPluginRecommended from 'eslint-plugin-prettier/recommended';
import reactPlugin from 'eslint-plugin-react';
import reactHooksPlugin from 'eslint-plugin-react-hooks';
import reactRefreshPlugin from 'eslint-plugin-react-refresh';
import { defineConfig } from 'eslint/config';
import globals from 'globals';
import { configs as tseslintConfigs } from 'typescript-eslint';

export default defineConfig(
  js.configs.recommended,
  tseslintConfigs.recommended,
  tseslintConfigs.strictTypeChecked,
  tseslintConfigs.stylisticTypeChecked,

  importPlugin.flatConfigs.recommended,
  importPlugin.flatConfigs.typescript,
  reactPlugin.configs.flat.recommended,
  reactPlugin.configs.flat['jsx-runtime'],
  perfectionistConfigs['recommended-natural'],
  eslintConfigPrettier,
  prettierPluginRecommended,
  reactRefreshPlugin.configs.vite,
  {
    languageOptions: {
      globals: {
        ...globals.browser,
      },
      parserOptions: {
        projectService: {
          allowDefaultProject: ['eslint.config.mjs'],
        },
        tsconfigRootDir: import.meta.dirname,
      },
    },

    plugins: {
      'better-tailwindcss': tailwindcssPlugin,
      'react-hooks': reactHooksPlugin,
    },
    rules: {
      '@typescript-eslint/restrict-template-expressions': ['off', { allowNumber: true }],
      ...reactHooksPlugin.configs.recommended.rules,
      ...tailwindcssPlugin.configs['recommended-error'].rules,
      'better-tailwindcss/enforce-consistent-line-wrapping': ['off'],
      'import/no-extraneous-dependencies': [
        'error',
        {
          devDependencies: [
            '**/*.test.{ts,tsx,js,jsx}',
            '**/*.spec.{ts,tsx,js,jsx}',
            '**/testUtils.ts',
            '**/vite.config.{ts,js}',
            '**/vitest.config.{ts,js}',
            '**/eslint.config.{js,mjs}',
            '**/*.config.{ts,js,mjs}',
            'vite.config.ts',
          ],
          includeTypes: true,
        },
      ],
      'no-console': ['warn', { allow: ['error'] }],
      'object-shorthand': 'error',
      'perfectionist/sort-interfaces': 'off',
      'perfectionist/sort-modules': 'off',
      'perfectionist/sort-objects': 'off',
      'perfectionist/sort-union-types': 'off',
      'prefer-template': 'error',
      'prettier/prettier': [
        'error',
        {
          arrowParens: 'always',
          bracketSpacing: true,
          printWidth: 80,
          semi: true,
          singleQuote: true,
          tabWidth: 2,
          trailingComma: 'all',
          useTabs: false,
        },
      ],
      'react/no-children-prop': ['error', { allowFunctions: true }],
      'react/prop-types': 'off',
      'react/react-in-jsx-scope': 'off',
    },
    settings: {
      'better-tailwindcss': {
        entryPoint: 'src/index.css',
      },
      'import/parsers': {
        '@typescript-eslint/parser': ['.ts', '.tsx'],
      },
      'import/resolver': {
        node: {
          extensions: ['.js', '.jsx', '.ts', '.tsx'],
        },
        typescript: {
          alwaysTryTypes: true,
        },
      },
      react: {
        version: 'detect',
      },
    },
  },
  {
    files: ['*.js', '*.mjs', '*.cjs'],
    ...tseslintConfigs.disableTypeChecked,
  },
  {
    files: ['vite.config.ts'],
    rules: {
      // vite.config uses dev dependencies
      'import/no-extraneous-dependencies': 'off',
    },
  },
  {
    ignores: ['dist/**', 'routeTree.gen.ts', 'public/sw.js'],
  },
  {
    rules: {
      'prettier/prettier': [
        'error',
        {
          arrowParens: 'always',
          bracketSpacing: true,
          printWidth: 120,
          semi: true,
          singleQuote: true,
          tabWidth: 2,
          trailingComma: 'all',
          useTabs: false,
        },
      ],
    },
  },
);
