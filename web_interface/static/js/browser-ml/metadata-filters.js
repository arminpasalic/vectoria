const FILTER_TYPES = new Set(['exact', 'category', 'text', 'number', 'date', 'boolean', 'range']);

export function resolveMetadataValue(document, field) {
    if (!document || typeof document !== 'object') return undefined;
    if (Object.prototype.hasOwnProperty.call(document, field)) return document[field];
    if (document.metadata && Object.prototype.hasOwnProperty.call(document.metadata, field)) {
        return document.metadata[field];
    }
    if (document.data && Object.prototype.hasOwnProperty.call(document.data, field)) {
        return document.data[field];
    }
    return undefined;
}

export function normalizeMetadataFilters(filters = {}, documents = null) {
    if (filters === null || filters === undefined) return {};
    if (!isPlainObject(filters)) {
        throw new Error('Metadata filters must be an object keyed by metadata field.');
    }

    const normalized = {};
    for (const [field, rawFilter] of Object.entries(filters)) {
        const name = String(field || '').trim();
        if (!name) throw new Error('Metadata filter field names cannot be empty.');
        normalized[name] = normalizeFilterConfig(name, rawFilter);
    }

    if (Array.isArray(documents) && documents.length > 0) {
        for (const field of Object.keys(normalized)) {
            const exists = documents.some(document => resolveMetadataValue(document, field) !== undefined);
            if (!exists) throw new Error(`Unknown metadata filter field: ${field}`);
        }
    }

    return normalized;
}

export function mergeMetadataFilters(persistentFilters = {}, inlineFilters = {}) {
    return {
        ...(persistentFilters || {}),
        ...(inlineFilters || {})
    };
}

export function matchesMetadataFilters(document, normalizedFilters = {}) {
    return Object.entries(normalizedFilters).every(([field, config]) => {
        const actual = resolveMetadataValue(document, field);
        if (actual === undefined || actual === null) return false;
        return matchesFilterValue(actual, config);
    });
}

export function createMetadataFilterScope(documents, filters = {}) {
    const source = Array.isArray(documents) ? documents : [];
    const activeFields = Object.keys(filters || {});
    const applied = activeFields.length > 0;
    const indices = [];
    const matchedDocuments = [];

    source.forEach((document, index) => {
        if (!applied || matchesMetadataFilters(document, filters)) {
            indices.push(index);
            matchedDocuments.push(document);
        }
    });

    return {
        filters,
        activeFields,
        applied,
        indices,
        documents: matchedDocuments,
        matchedDocuments: matchedDocuments.length,
        totalDocuments: source.length,
        predicate: applied ? document => matchesMetadataFilters(document, filters) : null
    };
}

export function serializeMetadataFilterScope(scope) {
    return {
        applied: Boolean(scope?.applied),
        active_filters: scope?.activeFields || [],
        matched_documents: scope?.matchedDocuments ?? 0,
        total_documents: scope?.totalDocuments ?? 0,
        metadata_filters: scope?.filters || {}
    };
}

function normalizeFilterConfig(field, rawFilter) {
    if (Array.isArray(rawFilter)) {
        if (!rawFilter.length) throw new Error(`Metadata filter "${field}" cannot use an empty array.`);
        return { type: 'category', value: rawFilter };
    }

    if (isScalar(rawFilter)) {
        if (typeof rawFilter === 'string' && !rawFilter.trim()) {
            throw new Error(`Metadata filter "${field}" cannot be empty.`);
        }
        return {
            type: typeof rawFilter === 'boolean' ? 'boolean' : 'exact',
            value: rawFilter
        };
    }

    if (!isPlainObject(rawFilter)) {
        throw new Error(`Metadata filter "${field}" has an unsupported value.`);
    }

    if ('conditions' in rawFilter || 'range' in rawFilter) {
        if (rawFilter.range && Object.keys(rawFilter.range).length) {
            return normalizeTypedConfig(field, {
                type: rawFilter.type === 'date'
                    ? 'date'
                    : rawFilter.type === 'number'
                        ? 'number'
                        : 'range',
                value: rawFilter.range
            });
        }
        const conditions = Array.isArray(rawFilter.conditions)
            && rawFilter.type !== 'category'
            && rawFilter.conditions.length === 1
            ? rawFilter.conditions[0]
            : rawFilter.conditions;
        return normalizeTypedConfig(field, {
            type: FILTER_TYPES.has(rawFilter.type) ? rawFilter.type : 'category',
            value: conditions
        });
    }

    if ('type' in rawFilter || 'value' in rawFilter) {
        return normalizeTypedConfig(field, rawFilter);
    }

    if ('min' in rawFilter || 'max' in rawFilter) {
        return normalizeTypedConfig(field, { type: 'range', value: rawFilter });
    }

    throw new Error(`Metadata filter "${field}" has an unsupported object shape.`);
}

function normalizeTypedConfig(field, rawConfig) {
    const type = String(rawConfig.type || 'exact').toLowerCase();
    if (!FILTER_TYPES.has(type)) {
        throw new Error(`Metadata filter "${field}" has unsupported type "${type}".`);
    }
    const value = rawConfig.value;
    if (value === undefined || value === null || value === '') {
        throw new Error(`Metadata filter "${field}" is missing a value.`);
    }

    if (type === 'category') {
        const values = Array.isArray(value) ? value : [value];
        if (!values.length) throw new Error(`Metadata filter "${field}" cannot use an empty category list.`);
        return { type, value: values };
    }

    if (type === 'number' || type === 'date' || type === 'range') {
        if (isPlainObject(value)) {
            if (value.min === undefined && value.max === undefined) {
                throw new Error(`Metadata filter "${field}" range requires min or max.`);
            }
            validateRangeBounds(field, value, type);
            return { type, value: { ...value } };
        }
        if (type === 'number' && !Number.isFinite(Number(value))) {
            throw new Error(`Metadata filter "${field}" requires a numeric value.`);
        }
        if (type === 'date' && !Number.isFinite(Date.parse(value))) {
            throw new Error(`Metadata filter "${field}" requires a valid date.`);
        }
    }

    if (type === 'boolean' && parseBoolean(value) === null) {
        throw new Error(`Metadata filter "${field}" requires a boolean value.`);
    }

    return { type, value };
}

function validateRangeBounds(field, range, type) {
    const bounds = [range.min, range.max].filter(value => value !== undefined && value !== null && value !== '');
    if (type === 'number' && bounds.some(value => !Number.isFinite(Number(value)))) {
        throw new Error(`Metadata filter "${field}" requires numeric range bounds.`);
    }
    if (type === 'date' && bounds.some(value => !Number.isFinite(Date.parse(value)))) {
        throw new Error(`Metadata filter "${field}" requires valid date range bounds.`);
    }
    if (type === 'range') {
        const numeric = bounds.every(value => Number.isFinite(Number(value)));
        const dates = bounds.every(value => Number.isFinite(Date.parse(value)));
        if (!numeric && !dates) {
            throw new Error(`Metadata filter "${field}" range must contain numbers or dates.`);
        }
    }
}

function matchesFilterValue(actual, config) {
    const actualValues = Array.isArray(actual) ? actual : [actual];

    if (config.type === 'category') {
        return actualValues.some(candidate =>
            config.value.some(expected => normalizedEquals(candidate, expected))
        );
    }
    if (config.type === 'text') {
        const expected = normalizeString(config.value);
        return actualValues.some(candidate => normalizeString(candidate).includes(expected));
    }
    if (config.type === 'boolean') {
        const expected = parseBoolean(config.value);
        return actualValues.some(candidate => parseBoolean(candidate) === expected);
    }
    if (config.type === 'number') {
        if (isPlainObject(config.value)) {
            return actualValues.some(candidate => matchesNumericRange(candidate, config.value));
        }
        return actualValues.some(candidate => Number(candidate) === Number(config.value));
    }
    if (config.type === 'date') {
        if (isPlainObject(config.value)) {
            return actualValues.some(candidate => matchesDateRange(candidate, config.value));
        }
        const expected = Date.parse(config.value);
        return actualValues.some(candidate => Date.parse(candidate) === expected);
    }
    if (config.type === 'range') {
        return actualValues.some(candidate => matchesAutomaticRange(candidate, config.value));
    }
    return actualValues.some(candidate => normalizedEquals(candidate, config.value));
}

function matchesNumericRange(actual, range) {
    const value = Number(actual);
    if (!Number.isFinite(value)) return false;
    return (range.min === undefined || value >= Number(range.min))
        && (range.max === undefined || value <= Number(range.max));
}

function matchesDateRange(actual, range) {
    const value = Date.parse(actual);
    if (!Number.isFinite(value)) return false;
    return (range.min === undefined || value >= Date.parse(range.min))
        && (range.max === undefined || value <= Date.parse(range.max));
}

function matchesAutomaticRange(actual, range) {
    const bounds = [range.min, range.max].filter(value => value !== undefined && value !== null && value !== '');
    if (bounds.every(value => Number.isFinite(Number(value))) && Number.isFinite(Number(actual))) {
        return matchesNumericRange(actual, range);
    }
    return matchesDateRange(actual, range);
}

function normalizedEquals(actual, expected) {
    if (typeof actual === 'number' || typeof expected === 'number') {
        const left = Number(actual);
        const right = Number(expected);
        if (Number.isFinite(left) && Number.isFinite(right)) return left === right;
    }
    return normalizeString(actual) === normalizeString(expected);
}

function normalizeString(value) {
    return String(value ?? '').trim().toLowerCase();
}

function parseBoolean(value) {
    if (value === true || value === 1) return true;
    if (value === false || value === 0) return false;
    const normalized = normalizeString(value);
    if (['true', '1', 'yes', 'y'].includes(normalized)) return true;
    if (['false', '0', 'no', 'n'].includes(normalized)) return false;
    return null;
}

function isPlainObject(value) {
    return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function isScalar(value) {
    return ['string', 'number', 'boolean'].includes(typeof value);
}
