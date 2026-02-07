# is-any-array

[![NPM version][npm-image]][npm-url]
[![build status][travis-image]][travis-url]
[![Test coverage][codecov-image]][codecov-url]
[![David deps][david-image]][david-url]
[![npm download][download-image]][download-url]

Check if toString.call ends with Array

## Installation

`$ npm install is-any-array`

## [API Documentation](https://cheminfo-js.github.io/is-any-array/)

## Example

```js
const isAnyArray = require('is-any-array');

  isAnyArray(1); // false
  isAnyArray('ab'); // false
  isAnyArray({ a: 1 }); // false

  isAnyArray([1, 2, 3]); // true
  isAnyArray(new Uint16Array(2))) // true;
```

## License

[MIT](./LICENSE)

[npm-image]: https://img.shields.io/npm/v/is-any-array.svg?style=flat-square
[npm-url]: https://www.npmjs.com/package/is-any-array
[travis-image]: https://img.shields.io/travis/cheminfo-js/is-any-array/master.svg?style=flat-square
[travis-url]: https://travis-ci.org/cheminfo-js/is-any-array
[codecov-image]: https://img.shields.io/codecov/c/github/cheminfo-js/is-any-array.svg?style=flat-square
[codecov-url]: https://codecov.io/gh/cheminfo-js/is-any-array
[david-image]: https://img.shields.io/david/cheminfo-js/is-any-array.svg?style=flat-square
[david-url]: https://david-dm.org/cheminfo-js/is-any-array
[download-image]: https://img.shields.io/npm/dm/is-any-array.svg?style=flat-square
[download-url]: https://www.npmjs.com/package/is-any-array
