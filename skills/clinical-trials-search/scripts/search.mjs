#!/usr/bin/env node

/**
 * ClinicalTrials Search via Valyu API
 * Full-text search across ClinicalTrials with semantic search capabilities
 */

import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs';
import { homedir } from 'os';
import { join } from 'path';

const VALYU_API_BASE = 'https://api.valyu.ai/v1';
const CONFIG_DIR = join(homedir(), '.valyu');
const CONFIG_FILE = join(CONFIG_DIR, 'config.json');

/**
 * Get API key from multiple sources (in order of priority):
 * 1. Environment variable (VALYU_API_KEY)
 * 2. Config file (~/.valyu/config.json)
 */
function getApiKey() {
  if (process.env.VALYU_API_KEY) {
    return process.env.VALYU_API_KEY;
  }

  if (existsSync(CONFIG_FILE)) {
    try {
      const config = JSON.parse(readFileSync(CONFIG_FILE, 'utf-8'));
      if (config.apiKey) {
        return config.apiKey;
      }
    } catch (e) {
      // Ignore parse errors
    }
  }

  return null;
}

/**
 * Save API key to config file
 */
function saveApiKey(apiKey) {
  if (!existsSync(CONFIG_DIR)) {
    mkdirSync(CONFIG_DIR, { recursive: true });
  }

  let config = {};
  if (existsSync(CONFIG_FILE)) {
    try {
      config = JSON.parse(readFileSync(CONFIG_FILE, 'utf-8'));
    } catch (e) {
      // Start fresh if parse fails
    }
  }

  config.apiKey = apiKey;
  writeFileSync(CONFIG_FILE, JSON.stringify(config, null, 2));
  return true;
}

/**
 * Return setup required response
 */
function setupRequiredResponse() {
  return {
    success: false,
    setup_required: true,
    message: "Valyu API key not configured. Get your free API key ($10 credits) at https://platform.valyu.ai"
  };
}

/**
 * Search ClinicalTrials via Valyu API
 */
async function searchClinicalTrials(query, maxResults = 10) {
  const apiKey = getApiKey();

  if (!apiKey) {
    return setupRequiredResponse();
  }

  try {
    const response = await fetch(`${VALYU_API_BASE}/search`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': apiKey
      },
      body: JSON.stringify({
        query: query,
        search_type: 'proprietary',
        included_sources: ['valyu/valyu-clinical-trials'],
        limit: maxResults
      })
    });

    const data = await response.json();

    if (!response.ok) {
      return {
        success: false,
        error: data.detail || data.message || `HTTP ${response.status}`,
        status: response.status
      };
    }

    return {
      success: true,
      type: 'clinical_trials_search',
      query: query,
      result_count: data.results?.length || 0,
      results: data.results || [],
      cost: data.cost || 0
    };

  } catch (error) {
    return {
      success: false,
      error: error.message
    };
  }
}

/**
 * Setup command - save API key
 */
function setup(apiKey) {
  if (!apiKey) {
    return {
      success: false,
      error: "API key required. Usage: search setup <api-key>"
    };
  }

  saveApiKey(apiKey);
  return {
    success: true,
    type: 'setup',
    message: "API key saved to ~/.valyu/config.json"
  };
}

// Main CLI handler
const [,, command, ...args] = process.argv;

(async () => {
  let result;

  if (command === 'setup') {
    result = setup(args[0]);
  } else {
    // Treat first arg as query, second as maxResults
    const query = command || '';
    const maxResults = args[0] ? parseInt(args[0], 10) : 10;

    if (!query) {
      result = {
        success: false,
        error: "Query required. Usage: search <query> [maxResults]"
      };
    } else {
      result = await searchClinicalTrials(query, maxResults);
    }
  }

  console.log(JSON.stringify(result, null, 2));
  process.exit(result.success ? 0 : 1);
})();
