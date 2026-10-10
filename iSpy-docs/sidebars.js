// @ts-check

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.

 @type {import('@docusaurus/plugin-content-docs').SidebarsConfig}
 */
const sidebars = {
  tutorialSidebar: [
    'intro',
    'architecture',
    {
      type: 'category',
      label: 'Getting Started',
      items: [
        'installation',
        'getting-started',
        'configuration',
        'cli-reference',
      ],
    },
    {
      type: 'category',
      label: 'Vision Setup',
      items: [
        'cameras-and-models',
        'vision-pipelines',
        'calibration',
        'web-ui-and-networking',
      ],
    },
    {
      type: 'category',
      label: 'Operations',
      items: [
        'datasets-and-recordings',
        'troubleshooting',
        'development-and-advanced-config',
      ],
    },
  ],
};

export default sidebars;
