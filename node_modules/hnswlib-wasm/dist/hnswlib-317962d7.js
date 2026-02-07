var hnswlib = (() => {
  var _scriptDir = import.meta.url;
  
  return (
function(hnswlib = {})  {

// include: shell.js
// The Module object: Our interface to the outside world. We import
// and export values on it. There are various ways Module can be used:
// 1. Not defined. We create it here
// 2. A function parameter, function(Module) { ..generated code.. }
// 3. pre-run appended it, var Module = {}; ..generated code..
// 4. External script tag defines var Module.
// We need to check if Module already exists (e.g. case 3 above).
// Substitution will be replaced with actual code on later stage of the build,
// this way Closure Compiler will not mangle it (e.g. case 4. above).
// Note that if you want to run closure, and also to use Module
// after the generated code, you will need to define   var Module = {};
// before the code. Then that object will be used in the code, and you
// can continue to use Module afterwards as well.
var Module = typeof hnswlib != 'undefined' ? hnswlib : {};

// Set up the promise that indicates the Module is initialized
var readyPromiseResolve, readyPromiseReject;
Module['ready'] = new Promise(function(resolve, reject) {
  readyPromiseResolve = resolve;
  readyPromiseReject = reject;
});
["_main","getExceptionMessage","___get_exception_message","_free","___cpp_exception","___cxa_increment_exception_refcount","___cxa_decrement_exception_refcount","___thrown_object_from_unwind_exception","___getTypeName","__embind_initialize_bindings","_fflush","onRuntimeInitialized"].forEach((prop) => {
  if (!Object.getOwnPropertyDescriptor(Module['ready'], prop)) {
    Object.defineProperty(Module['ready'], prop, {
      get: () => abort('You are getting ' + prop + ' on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js'),
      set: () => abort('You are setting ' + prop + ' on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js'),
    });
  }
});

// --pre-jses are emitted after the Module integration code, so that they can
// refer to Module (if they choose; they can also define Module)


// Sometimes an existing Module object exists with properties
// meant to overwrite the default module functionality. Here
// we collect those properties and reapply _after_ we configure
// the current environment's defaults to avoid having to be so
// defensive during initialization.
var moduleOverrides = Object.assign({}, Module);
var thisProgram = './this.program';

// Determine the runtime environment we are in. You can customize this by
// setting the ENVIRONMENT setting at compile time (see settings.js).

var ENVIRONMENT_IS_WEB = true;
var ENVIRONMENT_IS_WORKER = false;
var ENVIRONMENT_IS_NODE = false;
var ENVIRONMENT_IS_SHELL = false;

if (Module['ENVIRONMENT']) {
  throw new Error('Module.ENVIRONMENT has been deprecated. To force the environment, use the ENVIRONMENT compile-time option (for example, -sENVIRONMENT=web or -sENVIRONMENT=node)');
}

// `/` should be present at the end if `scriptDirectory` is not empty
var scriptDirectory = '';
function locateFile(path) {
  if (Module['locateFile']) {
    return Module['locateFile'](path, scriptDirectory);
  }
  return scriptDirectory + path;
}

// Hooks that are implemented differently in different runtime environments.
var read_,
    readAsync,
    readBinary;

{
  if (typeof document != 'undefined' && document.currentScript) { // web
    scriptDirectory = document.currentScript.src;
  }
  // When MODULARIZE, this JS may be executed later, after document.currentScript
  // is gone, so we saved it, and we use it here instead of any other info.
  if (_scriptDir) {
    scriptDirectory = _scriptDir;
  }
  // blob urls look like blob:http://site.com/etc/etc and we cannot infer anything from them.
  // otherwise, slice off the final part of the url to find the script directory.
  // if scriptDirectory does not contain a slash, lastIndexOf will return -1,
  // and scriptDirectory will correctly be replaced with an empty string.
  // If scriptDirectory contains a query (starting with ?) or a fragment (starting with #),
  // they are removed because they could contain a slash.
  if (scriptDirectory.indexOf('blob:') !== 0) {
    scriptDirectory = scriptDirectory.substr(0, scriptDirectory.replace(/[?#].*/, "").lastIndexOf('/')+1);
  } else {
    scriptDirectory = '';
  }

  if (!(typeof window == 'object' || typeof importScripts == 'function')) throw new Error('not compiled for this environment (did you build to HTML and try to run it not on the web, or set ENVIRONMENT to something - like node - and run it someplace else - like on the web?)');

  // Differentiate the Web Worker from the Node Worker case, as reading must
  // be done differently.
  {
// include: web_or_worker_shell_read.js
read_ = (url) => {
    try {
      var xhr = new XMLHttpRequest();
      xhr.open('GET', url, false);
      xhr.send(null);
      return xhr.responseText;
    } catch (err) {
      var data = tryParseAsDataURI(url);
      if (data) {
        return intArrayToString(data);
      }
      throw err;
    }
  };

  readAsync = (url, onload, onerror) => {
    var xhr = new XMLHttpRequest();
    xhr.open('GET', url, true);
    xhr.responseType = 'arraybuffer';
    xhr.onload = () => {
      if (xhr.status == 200 || (xhr.status == 0 && xhr.response)) { // file URLs can return 0
        onload(xhr.response);
        return;
      }
      var data = tryParseAsDataURI(url);
      if (data) {
        onload(data.buffer);
        return;
      }
      onerror();
    };
    xhr.onerror = onerror;
    xhr.send(null);
  };

// end include: web_or_worker_shell_read.js
  }
}

var out = Module['print'] || console.log.bind(console);
var err = Module['printErr'] || console.warn.bind(console);

// Merge back in the overrides
Object.assign(Module, moduleOverrides);
// Free the object hierarchy contained in the overrides, this lets the GC
// reclaim data used e.g. in memoryInitializerRequest, which is a large typed array.
moduleOverrides = null;
checkIncomingModuleAPI();

// Emit code to handle expected values on the Module object. This applies Module.x
// to the proper local x. This has two benefits: first, we only emit it if it is
// expected to arrive, and second, by using a local everywhere else that can be
// minified.

if (Module['arguments']) Module['arguments'];legacyModuleProp('arguments', 'arguments_');

if (Module['thisProgram']) thisProgram = Module['thisProgram'];legacyModuleProp('thisProgram', 'thisProgram');

if (Module['quit']) Module['quit'];legacyModuleProp('quit', 'quit_');

// perform assertions in shell.js after we set up out() and err(), as otherwise if an assertion fails it cannot print the message
// Assertions on removed incoming Module JS APIs.
assert(typeof Module['memoryInitializerPrefixURL'] == 'undefined', 'Module.memoryInitializerPrefixURL option was removed, use Module.locateFile instead');
assert(typeof Module['pthreadMainPrefixURL'] == 'undefined', 'Module.pthreadMainPrefixURL option was removed, use Module.locateFile instead');
assert(typeof Module['cdInitializerPrefixURL'] == 'undefined', 'Module.cdInitializerPrefixURL option was removed, use Module.locateFile instead');
assert(typeof Module['filePackagePrefixURL'] == 'undefined', 'Module.filePackagePrefixURL option was removed, use Module.locateFile instead');
assert(typeof Module['read'] == 'undefined', 'Module.read option was removed (modify read_ in JS)');
assert(typeof Module['readAsync'] == 'undefined', 'Module.readAsync option was removed (modify readAsync in JS)');
assert(typeof Module['readBinary'] == 'undefined', 'Module.readBinary option was removed (modify readBinary in JS)');
assert(typeof Module['setWindowTitle'] == 'undefined', 'Module.setWindowTitle option was removed (modify setWindowTitle in JS)');
assert(typeof Module['TOTAL_MEMORY'] == 'undefined', 'Module.TOTAL_MEMORY has been renamed Module.INITIAL_MEMORY');
legacyModuleProp('read', 'read_');
legacyModuleProp('readAsync', 'readAsync');
legacyModuleProp('readBinary', 'readBinary');
legacyModuleProp('setWindowTitle', 'setWindowTitle');

assert(!ENVIRONMENT_IS_WORKER, "worker environment detected but not enabled at build time.  Add 'worker' to `-sENVIRONMENT` to enable.");

assert(!ENVIRONMENT_IS_NODE, "node environment detected but not enabled at build time.  Add 'node' to `-sENVIRONMENT` to enable.");

assert(!ENVIRONMENT_IS_SHELL, "shell environment detected but not enabled at build time.  Add 'shell' to `-sENVIRONMENT` to enable.");


// end include: shell.js
// include: preamble.js
// === Preamble library stuff ===

// Documentation for the public APIs defined in this file must be updated in:
//    site/source/docs/api_reference/preamble.js.rst
// A prebuilt local version of the documentation is available at:
//    site/build/text/docs/api_reference/preamble.js.txt
// You can also build docs locally as HTML or other formats in site/
// An online HTML version (which may be of a different version of Emscripten)
//    is up at http://kripken.github.io/emscripten-site/docs/api_reference/preamble.js.html

var wasmBinary;
if (Module['wasmBinary']) wasmBinary = Module['wasmBinary'];legacyModuleProp('wasmBinary', 'wasmBinary');
Module['noExitRuntime'] || true;legacyModuleProp('noExitRuntime', 'noExitRuntime');

if (typeof WebAssembly != 'object') {
  abort('no native wasm support detected');
}

// Wasm globals

var wasmMemory;

//========================================
// Runtime essentials
//========================================

// whether we are quitting the application. no code should run after this.
// set in exit() and abort()
var ABORT = false;

/** @type {function(*, string=)} */
function assert(condition, text) {
  if (!condition) {
    abort('Assertion failed' + (text ? ': ' + text : ''));
  }
}

// We used to include malloc/free by default in the past. Show a helpful error in
// builds with assertions.

// Memory management

var /** @type {!Int8Array} */
  HEAP8,
/** @type {!Uint8Array} */
  HEAPU8,
/** @type {!Int16Array} */
  HEAP16,
/** @type {!Uint16Array} */
  HEAPU16,
/** @type {!Int32Array} */
  HEAP32,
/** @type {!Uint32Array} */
  HEAPU32,
/** @type {!Float32Array} */
  HEAPF32,
/** @type {!Float64Array} */
  HEAPF64;

function updateMemoryViews() {
  var b = wasmMemory.buffer;
  Module['HEAP8'] = HEAP8 = new Int8Array(b);
  Module['HEAP16'] = HEAP16 = new Int16Array(b);
  Module['HEAP32'] = HEAP32 = new Int32Array(b);
  Module['HEAPU8'] = HEAPU8 = new Uint8Array(b);
  Module['HEAPU16'] = HEAPU16 = new Uint16Array(b);
  Module['HEAPU32'] = HEAPU32 = new Uint32Array(b);
  Module['HEAPF32'] = HEAPF32 = new Float32Array(b);
  Module['HEAPF64'] = HEAPF64 = new Float64Array(b);
}

assert(!Module['STACK_SIZE'], 'STACK_SIZE can no longer be set at runtime.  Use -sSTACK_SIZE at link time');

assert(typeof Int32Array != 'undefined' && typeof Float64Array !== 'undefined' && Int32Array.prototype.subarray != undefined && Int32Array.prototype.set != undefined,
       'JS engine does not provide full typed array support');

// If memory is defined in wasm, the user can't provide it, or set INITIAL_MEMORY
assert(!Module['wasmMemory'], 'Use of `wasmMemory` detected.  Use -sIMPORTED_MEMORY to define wasmMemory externally');
assert(!Module['INITIAL_MEMORY'], 'Detected runtime INITIAL_MEMORY setting.  Use -sIMPORTED_MEMORY to define wasmMemory dynamically');

// include: runtime_init_table.js
// In regular non-RELOCATABLE mode the table is exported
// from the wasm module and this will be assigned once
// the exports are available.
var wasmTable;

// end include: runtime_init_table.js
// include: runtime_stack_check.js
// Initializes the stack cookie. Called at the startup of main and at the startup of each thread in pthreads mode.
function writeStackCookie() {
  var max = _emscripten_stack_get_end();
  assert((max & 3) == 0);
  // If the stack ends at address zero we write our cookies 4 bytes into the
  // stack.  This prevents interference with the (separate) address-zero check
  // below.
  if (max == 0) {
    max += 4;
  }
  // The stack grow downwards towards _emscripten_stack_get_end.
  // We write cookies to the final two words in the stack and detect if they are
  // ever overwritten.
  HEAPU32[((max)>>2)] = 0x02135467;
  HEAPU32[(((max)+(4))>>2)] = 0x89BACDFE;
  // Also test the global address 0 for integrity.
  HEAPU32[0] = 0x63736d65; /* 'emsc' */
}

function checkStackCookie() {
  if (ABORT) return;
  var max = _emscripten_stack_get_end();
  // See writeStackCookie().
  if (max == 0) {
    max += 4;
  }
  var cookie1 = HEAPU32[((max)>>2)];
  var cookie2 = HEAPU32[(((max)+(4))>>2)];
  if (cookie1 != 0x02135467 || cookie2 != 0x89BACDFE) {
    abort('Stack overflow! Stack cookie has been overwritten at ' + ptrToString(max) + ', expected hex dwords 0x89BACDFE and 0x2135467, but received ' + ptrToString(cookie2) + ' ' + ptrToString(cookie1));
  }
  // Also test the global address 0 for integrity.
  if (HEAPU32[0] !== 0x63736d65 /* 'emsc' */) {
    abort('Runtime error: The application has corrupted its heap memory area (address zero)!');
  }
}

// end include: runtime_stack_check.js
// include: runtime_assertions.js
// Endianness check
(function() {
  var h16 = new Int16Array(1);
  var h8 = new Int8Array(h16.buffer);
  h16[0] = 0x6373;
  if (h8[0] !== 0x73 || h8[1] !== 0x63) throw 'Runtime error: expected the system to be little-endian! (Run with -sSUPPORT_BIG_ENDIAN to bypass)';
})();

// end include: runtime_assertions.js
var __ATPRERUN__  = []; // functions called before the runtime is initialized
var __ATINIT__    = []; // functions called during startup
var __ATPOSTRUN__ = []; // functions called after the main() is called

var runtimeInitialized = false;

function preRun() {
  if (Module['preRun']) {
    if (typeof Module['preRun'] == 'function') Module['preRun'] = [Module['preRun']];
    while (Module['preRun'].length) {
      addOnPreRun(Module['preRun'].shift());
    }
  }
  callRuntimeCallbacks(__ATPRERUN__);
}

function initRuntime() {
  assert(!runtimeInitialized);
  runtimeInitialized = true;

  checkStackCookie();

  
if (!Module["noFSInit"] && !FS.init.initialized)
  FS.init();
FS.ignorePermissions = false;
  callRuntimeCallbacks(__ATINIT__);
}

function postRun() {
  checkStackCookie();

  if (Module['postRun']) {
    if (typeof Module['postRun'] == 'function') Module['postRun'] = [Module['postRun']];
    while (Module['postRun'].length) {
      addOnPostRun(Module['postRun'].shift());
    }
  }

  callRuntimeCallbacks(__ATPOSTRUN__);
}

function addOnPreRun(cb) {
  __ATPRERUN__.unshift(cb);
}

function addOnInit(cb) {
  __ATINIT__.unshift(cb);
}

function addOnPostRun(cb) {
  __ATPOSTRUN__.unshift(cb);
}

// include: runtime_math.js
// https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Math/imul

// https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Math/fround

// https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Math/clz32

// https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Math/trunc

assert(Math.imul, 'This browser does not support Math.imul(), build with LEGACY_VM_SUPPORT or POLYFILL_OLD_MATH_FUNCTIONS to add in a polyfill');
assert(Math.fround, 'This browser does not support Math.fround(), build with LEGACY_VM_SUPPORT or POLYFILL_OLD_MATH_FUNCTIONS to add in a polyfill');
assert(Math.clz32, 'This browser does not support Math.clz32(), build with LEGACY_VM_SUPPORT or POLYFILL_OLD_MATH_FUNCTIONS to add in a polyfill');
assert(Math.trunc, 'This browser does not support Math.trunc(), build with LEGACY_VM_SUPPORT or POLYFILL_OLD_MATH_FUNCTIONS to add in a polyfill');

// end include: runtime_math.js
// A counter of dependencies for calling run(). If we need to
// do asynchronous work before running, increment this and
// decrement it. Incrementing must happen in a place like
// Module.preRun (used by emcc to add file preloading).
// Note that you can add dependencies in preRun, even though
// it happens right before run - run will be postponed until
// the dependencies are met.
var runDependencies = 0;
var runDependencyWatcher = null;
var dependenciesFulfilled = null; // overridden to take different actions when all run dependencies are fulfilled
var runDependencyTracking = {};

function getUniqueRunDependency(id) {
  var orig = id;
  while (1) {
    if (!runDependencyTracking[id]) return id;
    id = orig + Math.random();
  }
}

function addRunDependency(id) {
  runDependencies++;

  if (Module['monitorRunDependencies']) {
    Module['monitorRunDependencies'](runDependencies);
  }

  if (id) {
    assert(!runDependencyTracking[id]);
    runDependencyTracking[id] = 1;
    if (runDependencyWatcher === null && typeof setInterval != 'undefined') {
      // Check for missing dependencies every few seconds
      runDependencyWatcher = setInterval(function() {
        if (ABORT) {
          clearInterval(runDependencyWatcher);
          runDependencyWatcher = null;
          return;
        }
        var shown = false;
        for (var dep in runDependencyTracking) {
          if (!shown) {
            shown = true;
            err('still waiting on run dependencies:');
          }
          err('dependency: ' + dep);
        }
        if (shown) {
          err('(end of list)');
        }
      }, 10000);
    }
  } else {
    err('warning: run dependency added without ID');
  }
}

function removeRunDependency(id) {
  runDependencies--;

  if (Module['monitorRunDependencies']) {
    Module['monitorRunDependencies'](runDependencies);
  }

  if (id) {
    assert(runDependencyTracking[id]);
    delete runDependencyTracking[id];
  } else {
    err('warning: run dependency removed without ID');
  }
  if (runDependencies == 0) {
    if (runDependencyWatcher !== null) {
      clearInterval(runDependencyWatcher);
      runDependencyWatcher = null;
    }
    if (dependenciesFulfilled) {
      var callback = dependenciesFulfilled;
      dependenciesFulfilled = null;
      callback(); // can add another dependenciesFulfilled
    }
  }
}

/** @param {string|number=} what */
function abort(what) {
  if (Module['onAbort']) {
    Module['onAbort'](what);
  }

  what = 'Aborted(' + what + ')';
  // TODO(sbc): Should we remove printing and leave it up to whoever
  // catches the exception?
  err(what);

  ABORT = true;

  // Use a wasm runtime error, because a JS error might be seen as a foreign
  // exception, which means we'd run destructors on it. We need the error to
  // simply make the program stop.
  // FIXME This approach does not work in Wasm EH because it currently does not assume
  // all RuntimeErrors are from traps; it decides whether a RuntimeError is from
  // a trap or not based on a hidden field within the object. So at the moment
  // we don't have a way of throwing a wasm trap from JS. TODO Make a JS API that
  // allows this in the wasm spec.

  // Suppress closure compiler warning here. Closure compiler's builtin extern
  // defintion for WebAssembly.RuntimeError claims it takes no arguments even
  // though it can.
  // TODO(https://github.com/google/closure-compiler/pull/3913): Remove if/when upstream closure gets fixed.
  // See above, in the meantime, we resort to wasm code for trapping.
  //
  // In case abort() is called before the module is initialized, Module['asm']
  // and its exported '__trap' function is not available, in which case we throw
  // a RuntimeError.
  //
  // We trap instead of throwing RuntimeError to prevent infinite-looping in
  // Wasm EH code (because RuntimeError is considered as a foreign exception and
  // caught by 'catch_all'), but in case throwing RuntimeError is fine because
  // the module has not even been instantiated, even less running.
  if (runtimeInitialized) {
    ___trap();
  }
  /** @suppress {checkTypes} */
  var e = new WebAssembly.RuntimeError(what);

  readyPromiseReject(e);
  // Throw the error whether or not MODULARIZE is set because abort is used
  // in code paths apart from instantiation where an exception is expected
  // to be thrown when abort is called.
  throw e;
}

// include: memoryprofiler.js
// end include: memoryprofiler.js
// include: URIUtils.js
// Prefix of data URIs emitted by SINGLE_FILE and related options.
var dataURIPrefix = 'data:application/octet-stream;base64,';

// Indicates whether filename is a base64 data URI.
function isDataURI(filename) {
  // Prefix of data URIs emitted by SINGLE_FILE and related options.
  return filename.startsWith(dataURIPrefix);
}

// Indicates whether filename is delivered via file protocol (as opposed to http/https)
function isFileURI(filename) {
  return filename.startsWith('file://');
}

// end include: URIUtils.js
/** @param {boolean=} fixedasm */
function createExportWrapper(name, fixedasm) {
  return function() {
    var displayName = name;
    var asm = fixedasm;
    if (!fixedasm) {
      asm = Module['asm'];
    }
    assert(runtimeInitialized, 'native function `' + displayName + '` called before runtime initialization');
    if (!asm[name]) {
      assert(asm[name], 'exported native function `' + displayName + '` not found');
    }
    return asm[name].apply(null, arguments);
  };
}

// include: runtime_exceptions.js
// end include: runtime_exceptions.js
var wasmBinaryFile;
  wasmBinaryFile = 'data:application/octet-stream;base64,AGFzbQEAAAAB9QM8YAJ/fwBgAX8Bf2ABfwBgAn9/AX9gA39/fwF/YAN/f38AYAR/f39/AX9gBH9/f38AYAZ/f39/f38Bf2AFf39/f38Bf2AFf39/f38AYAN/f38BfWAAAGAGf39/f39/AGAIf39/f39/f38Bf2AAAX9gB39/f39/f38Bf2AHf39/f39/fwBgBX9+fn5+AGADf35/AX5gBX9/fn9/AGAFf39/f34Bf2AEf39/fwF+YAh/f39/f39/fwBgBH9+fn8AYAp/f39/f39/f39/AX9gBn9/f39+fgF/YAd/f39/f35+AX9gA39/fwF8YAZ/fH9/f38Bf2ADf35/AX9gDH9/f39/f39/f39/fwF/YAV/f39/fAF/YAR/f398AX9gBX9/f35+AX9gC39/f39/f39/f39/AX9gCn9/f39/f39/f38AYA9/f39/f39/f39/f39/f38AYA1/f39/f39/f39/f39/AGAJf39/f39/f39/AGAFf39/f38BfGAEf39/fwF9YAF8AXxgAnx/AXxgAn5/AX9gAn5+AXxgAX8BfmADf35/AGACf34AYAJ/fABgBH5+fn4Bf2ADfn5+AX9gAX8BfGACf38BfmACfn4BfWADf39+AGAEf39/fgF+YAV+f39/fwF/YAl/f39/f39/f38Bf2AEf39+fgACrAouA2Vudg1fZW12YWxfZGVjcmVmAAIDZW52GF9lbXZhbF9nZXRfbWV0aG9kX2NhbGxlcgADA2Vudg1fZW12YWxfaW5jcmVmAAIDZW52F19lbXZhbF9jYWxsX3ZvaWRfbWV0aG9kAAcDZW52GV9lbWJpbmRfcmVnaXN0ZXJfZnVuY3Rpb24AEQNlbnYWX2VtYmluZF9yZWdpc3Rlcl9jbGFzcwAmA2VudiJfZW1iaW5kX3JlZ2lzdGVyX2NsYXNzX2NvbnN0cnVjdG9yAA0DZW52H19lbWJpbmRfcmVnaXN0ZXJfY2xhc3NfZnVuY3Rpb24AJwNlbnYlX2VtYmluZF9yZWdpc3Rlcl9jbGFzc19jbGFzc19mdW5jdGlvbgAXA2VudhFfZW12YWxfdGFrZV92YWx1ZQADA2VudhBfZW12YWxfbmV3X2FycmF5AA8DZW52EV9lbXZhbF9uZXdfb2JqZWN0AA8DZW52El9lbXZhbF9uZXdfY3N0cmluZwABA2VudhNfZW12YWxfc2V0X3Byb3BlcnR5AAUDZW52GGVtc2NyaXB0ZW5fYXNtX2NvbnN0X2ludAAEA2VudgpzeW5jSWRiX2pzAAIDZW52IV9lbXZhbF9uZXdfYXJyYXlfZnJvbV9tZW1vcnlfdmlldwABA2VudhNfZW12YWxfZ2V0X3Byb3BlcnR5AAMDZW52CV9lbXZhbF9hcwAcA2VudhZfZW12YWxfcnVuX2Rlc3RydWN0b3JzAAIDZW52El9lbXZhbF9jYWxsX21ldGhvZAAoA2Vudg1fX2Fzc2VydF9mYWlsAAcDZW52FV9lbWJpbmRfcmVnaXN0ZXJfdm9pZAAAA2VudhVfZW1iaW5kX3JlZ2lzdGVyX2Jvb2wACgNlbnYYX2VtYmluZF9yZWdpc3Rlcl9pbnRlZ2VyAAoDZW52Fl9lbWJpbmRfcmVnaXN0ZXJfZmxvYXQABQNlbnYbX2VtYmluZF9yZWdpc3Rlcl9zdGRfc3RyaW5nAAADZW52HF9lbWJpbmRfcmVnaXN0ZXJfc3RkX3dzdHJpbmcABQNlbnYWX2VtYmluZF9yZWdpc3Rlcl9lbXZhbAAAA2VudhxfZW1iaW5kX3JlZ2lzdGVyX21lbW9yeV92aWV3AAUDZW52FWVtc2NyaXB0ZW5fbWVtY3B5X2JpZwAFFndhc2lfc25hcHNob3RfcHJldmlldzEIZmRfd3JpdGUABgNlbnYWZW1zY3JpcHRlbl9yZXNpemVfaGVhcAABA2VudhBfX3N5c2NhbGxfb3BlbmF0AAYDZW52EV9fc3lzY2FsbF9mY250bDY0AAQDZW52D19fc3lzY2FsbF9pb2N0bAAEFndhc2lfc25hcHNob3RfcHJldmlldzEHZmRfcmVhZAAGFndhc2lfc25hcHNob3RfcHJldmlldzEIZmRfY2xvc2UAARZ3YXNpX3NuYXBzaG90X3ByZXZpZXcxEWVudmlyb25fc2l6ZXNfZ2V0AAMWd2FzaV9zbmFwc2hvdF9wcmV2aWV3MQtlbnZpcm9uX2dldAADA2VudgpzdHJmdGltZV9sAAkDZW52EF9fc3lzY2FsbF9zdGF0NjQAAwNlbnYFYWJvcnQADANlbnYiX190aHJvd19leGNlcHRpb25fd2l0aF9zdGFja190cmFjZQACA2VudhdfZW1iaW5kX3JlZ2lzdGVyX2JpZ2ludAARFndhc2lfc25hcHNob3RfcHJldmlldzEHZmRfc2VlawAJA8wHygcMAgICAwwCAgAMDAMBAgMBCykBAwEBCwECAwEEAQIEAwAFAAMABQAFBwAKCQEBAQIGBAoNBQcAAAUEBwoHCgcGAAMAAQAABQABAQAKAAECAQ8AAg8FAAIPAwEEBQIDCgADBQUFCAIAAgUAAAABDAIDCwEBAQILAQMHCgAKAAMBAQIAAwEGAgMCAgEHBQAFCgABAQIACgUJDQcFAQcAAAIBAAEMDAQEBCoEBAABAwEBBAYCAgQTAQ8rCRAFAQcsCgQdAAMDAQIDAAEYGC0BBAwDAwMTBAEeHgEBAQQuAgECAAQUBwQFBAEBAwQDAQECAgACAQADAQEFAC8BAQICAAIABQUBAwEBAAEABQAEAQEBAQQCAgEMAwMEFAcBAAIBAgIBAgQFBwUABQMDAQEBBQECAAECAAIwAQASEjEyMzQSABIYEhISBzU2BjcEBAMDBAEDBgQDBgIAAQY4BwkHBQQJBwUECAEQAwACCAEFGQYHCBYIBggGCBYICh8LCBwIBwgPBAQDCAEQCAMFGQgICAgICh8ICAgECQEBCQcJBBEIFQkVICEEBhEaIgkECQEJEQgVCRUgIREaIgkEAAAOAQgICA0IDQgKCQ4OCAgIDQgNCAoJDhANEAIBAAAAAQADECMABQUQBQABAQMAECMAEAUBAAEDGyQlBAgbJCUECAQNDQEMAgACAgEFAgIBAgADAgICAgQGBgYDBAMEBgQJAQIDBAMEBgQJDgkJAg4EDgkJAQEJAQ4OCQEODgkBAgECAQEAAAAAAAAAAAECAQIAAQIBAgECAQIBAgECAQIBAgECAQIBAgECAgEFAwAFBQAFAQEFBQACDAIWFgIMAQMBAgIABQwCAwUDAQUABQcCAQICAgUCAwADAwMFFwEKBQUEAwADFwEKAAAAAwMDAwUEBAABBQEFBQQEAQIFAAIBDAwCAAwMDAIMAgQEBAMFBwcHBwQDBwoNCgoKDQ0NAQEBAQICAQEDAQE5AAIMDw8PBgIDAwMBAwUEAQACAgEBAwMBAQAFEAIDAAMBAQMBAwQEBAEDAAEDAwMAAQUBAwQIAAMAAAMDAAAAAQECBgMEAQMDAwkDAwMDAwAAAQAEBgEBAwMFAwMDAAYBAQQBAwAABwAAAAAAAAAAAAAAAAYAAwECAAAAAAAAAwAAAAAAAAAAAAAAAAADAAACAwAAAAQAAAAAAAAAAAMAAAAAAAQAAAAAAAYAAAADAwMDAAAAAAADAAAABAAAAAAAAAMDAwAAAAAABAAAAAAAAAMAAQAABQAEAAEFDwIBCREQOhk7BAUBcADSBQUHAQGAAoCAAg0DAQACBhcEfwFB8IYHC38BQQALfwFBAAt/AUEACwftBB0GbWVtb3J5AgARX193YXNtX2NhbGxfY3RvcnMALg9fX2NwcF9leGNlcHRpb24EAAZtYWxsb2MA8gEEZnJlZQDzARlfX2luZGlyZWN0X2Z1bmN0aW9uX3RhYmxlAQANX19nZXRUeXBlTmFtZQDPARtfZW1iaW5kX2luaXRpYWxpemVfYmluZGluZ3MA0AEQX19lcnJub19sb2NhdGlvbgDkAQZmZmx1c2gAhQIGX190cmFwAK8FFWVtc2NyaXB0ZW5fc3RhY2tfaW5pdACWBhllbXNjcmlwdGVuX3N0YWNrX2dldF9mcmVlAJcGGWVtc2NyaXB0ZW5fc3RhY2tfZ2V0X2Jhc2UAmAYYZW1zY3JpcHRlbl9zdGFja19nZXRfZW5kAJkGCXN0YWNrU2F2ZQDvBwxzdGFja1Jlc3RvcmUA8AcKc3RhY2tBbGxvYwDxBxxlbXNjcmlwdGVuX3N0YWNrX2dldF9jdXJyZW50AO8HIl9fY3hhX2RlY3JlbWVudF9leGNlcHRpb25fcmVmY291bnQA6AUiX19jeGFfaW5jcmVtZW50X2V4Y2VwdGlvbl9yZWZjb3VudADsBQ5fX2N4YV9kZW1hbmdsZQCaBiVfX3Rocm93bl9vYmplY3RfZnJvbV91bndpbmRfZXhjZXB0aW9uAO0HF19fZ2V0X2V4Y2VwdGlvbl9tZXNzYWdlAO4HDGR5bkNhbGxfamlqaQDyBw5keW5DYWxsX3ZpaWppaQDzBw5keW5DYWxsX2lpaWlpagD0Bw9keW5DYWxsX2lpaWlpamoA9QcQZHluQ2FsbF9paWlpaWlqagD2BwnFCgEAQQEL0QWOBjk2Ojs8PT4/QEFCOzxDRD9AQUVGR0hJSktMTU5PUFFSU1RVVldYWVpBW0BcXV5fYGFQUWJjZFNlT2ZnaGlqa2xtbm9wcUFyc3R1dkFAd3h5WXo1e3x9fn+AAX2BAYIBgwGEAYUBhgGHAYsGowGkAZ4BLy8wMTQ4nwGgAaEBpAGiAZ8BoAGhAaIBpQGmAacBqQGqAa0BrgG4AbwBvQG+AcAB0QHbAeEB4gHuAe8BgAKBAoIChgKkAYsCjAKNAo4CjwKQAtsB2wGRApQClQKWApcClgKZApsCmgKcAqcCqQKoAqoCvAK/AskCxQLGAscCyALBAsMCxAKxAcoCywLMAqwBzQLOAs8CL9wCpAGiAdoC2QXaBdsF2wLdAqAB3wLgAvkCgAPzAS+9BL8E9wT5BPwE/gSABYIFhAWGBYgFigWMBY4FkAWSBbgEuQS+BMsEzATNBM4EzwTQBP0B0QTSBNMErgTXBNgE2gTcBN0E2wHfBOAE6ATpBOwE7QTuBPAE8wTqBOsEnwG1Au8E8QT0BKQBogGiAcAEwQTCBMMExATFBMYExwT9AcgEyQTKBKIB1ATUBNUE2QHZAdYE2QGiAeEE4gTVBNsB2wHjBOQEogHhBOIE1QTbAdsB4wTkBKIB5QTmBNUE2wHbAecE5ASiAeUE5gTVBNsB2wHnBOQEpAGiAYkDigOMA6QBogGNA44DkAOiAZEDlwOdA58DoQOhA6MDpQOpA6sDrQOiAbIDtQO5A7oDuwO7A7wDvQPAA8EDwgOiAcQDxwPNA84DzwPQA9UD1wOiAdkD2wPeA98D4APhA+QD5gOkAaIB6wPsA+0D7gPwA/ID9QP2BPsE/wSLBY8FgwWHBaQBogHrA/cD+AP5A/sD/QOABPgE/QSBBY0FkQWFBYkFlAWTBYEElAWTBYMEogGFBIUEhgSGBIYEhwTbAYgEiASiAYUEhQSGBIYEhgSHBNsBiASIBKIBiQSJBIYEhgSGBIoE2wGIBIgEogGJBIkEhgSGBIYEigTbAYgEiASiAYwEkQSiAZgEmwSiAaEEpQSiAaYEqgSiAasErASNAqIBqwStBI0CpAGmBfIFpgXaAaoFrAW0BboFuwW8Bb0F3AKiAd0F3gWiAd8F4AXeBd0C5wUv7gXvBaQBogEvL/QFogH2BYcGhAb5BaIBhgaDBvoFogGFBoAG/AWiAf0FogGJBqIBigaiAYgGjQagAd0CjQaNBo0G3QKiAY8G2gHaAdoBmALKBo0CzAakAaIBpgXNBqIB0AbRBqIB0gaiAd4G4AbhBuIG4wbkBqIB+AaiAfwGogH9BqIB/gaiAf8GogGAB6IBggeiAYMHogGEB6IBhQeiAYYHogGIB6IBjAeiAY0HogGOB6IBjweiAZAHogGRB6IBkweiAZQHogGVB5YHogGXB5gHogGZB5YHogGaB5sHogGcB6IBngeiAZ8HogGhB6IBogeiAaUHogGmB6IBpweiAakHogGqB6IBqweiAawHogGtB6IBrgeiAa8HsAeiAbIHogGzB6IBtAeiAbUHtgeiAbgHuQeiAbsHugeiAbwHogG+B6IBvweiAcAHsAeiAa8HsAeiAa8HogHBB8IHwwfEB8UHxgeiAccHogHIB7YHogGtB6IByQeiAcoHygfLB8wHogHNB6IBzweiAdAHogHKB8oH0QfSB6IB0weiAdQHogHVB9YH1wfYB9kHogHaB6IB2weiAdwHogHeB6IBygfKB98H4AeiAdUH4QfiB6IB4weiAeQH5QfnB6IB5AfoB+oHogHsB6IBCoDTEMoHzwIBAn9B8IYHJAJB8IYDJAEjAEEQayIAJAACQCAAQQxqIABBCGoQJg0AQfDxAiAAKAIMQQJ0QQRqEPIBIgE2AgAgAUUNACAAKAIIEPIBIgEEQEHw8QIoAgAgACgCDEECdGpBADYCAEHw8QIoAgAgARAnRQ0BC0Hw8QJBADYCAAsgAEEQaiQAQeTjAkEAOgAAQeDjAkEANgIAQejjAkEBNgIAQfDjAkEQEKgFIgA2AgBB9OMCQo6AgICAgoCAgH83AgAgAEGmCykAADcABiAAQaALKQAANwAAIABBADoADkH84wJB7AA2AgBBgOQCQQA2AgAQOEGA5AJBlOQCKAIANgIAQZTkAkH84wI2AgBBmOQCQYIBNgIAQZzkAkEANgIAENEBQZzkAkGU5AIoAgA2AgBBlOQCQZjkAjYCAEHU7QJB3OwCNgIAQYztAkEqNgIACwMAAQsUAEHk4wItAAAEQEHg4wIoAgAaCwsbACMAIQAGQEHo4wIoAgAQABkgACQAEPAFAAsLhQEBA38gARDjASICQfD///8HSQRAAkACQCACQQtPBEAgAkEPckEBaiIDEKgFIQQgACADQYCAgIB4cjYCCCAAIAQ2AgAgACACNgIEIAIgBGohAwwBCyAAIAI6AAsgACACaiEDIAAhBCACRQ0BCyAEIAEgAhDTARoLIANBADoAACAADwsQMwALCQBBiSMQnAEACxkAQfvjAiwAAEEASARAQfDjAigCABDzAQsLqQIBAn8jAEEQayIBJABB7eMCLQAABEBBjykQ4AELQe7jAiAAOgAAQejjAigCAEEBRwRAQdU7EOABQejjAigCACEABkBBiOQCLQAAQQFxRQRAQQJB/NMAEAEhAkGI5AJBAToAAEGE5AIgAjYCAAtBhOQCKAIAIQJBARACIAFBATYCCCACIABBtR0gAUEIahADGSABJAAGQEEBEAAZIAEkABDwBQALCQALBkBBARAAGSABJAAQ8AUACwZAQejjAigCABAAGSABJAAGQEEBEAAZIAEkABDwBQALCQALQejjAkEBNgIABkBBABAAGSABJAAQ8AUACwsCQEHk4wItAABFDQBB4OMCKAIAGkHk4wJBADoAAEHt4wItAABFDQBB+ygQ4AELIAFBEGokAAv2BQIGfwJ9IwBBIGsiAiQAIABBADYCCCAAQgA3AgACQCABKAIEIgYgASgCACIBRg0ABkAgBiABayIEQQBIBEAQNwALIAQQqAUhAxkgAiQAIAAoAgAiAQRAIAAgATYCBCABEPMBCwZACQEHACEAIAIkAEHkhgNBgAg2AgBB4IYDQQA2AgAgABCVBgJAQeiGAygCAEEBRgRAIAAQ6QUhAEHt4wItAAAEQCACIAAgACgCACgCCBEBADYCAEGazwAgAhDYAQtBCBDkBSEDIAAgACgCACgCCBEBACEAIAJBAToAHwZAIAJBBGogABAyIQAgAkEBOgAeBkAgAiAAQZjIABDMBSIBKAIINgIYIAIgASkCADcDECABQgA3AgAgAUEANgIIIAJBAToAHQZAIAMgAkEQahDDBSEBIAJBADoAHSABQayQAkEBEOYFDAQZIAIkACACLQAdIQEgAiwAG0EASARAIAIoAhAQ8wELIAIgAUEBcToAHgkACwAZIAIkACACLQAeIQEgACwAC0EASARAIAAoAgAQ8wELIAIgAUEBcToAHwkACwAZIAIkACACLQAfBEAgAxDlBQsGQBDqBRkgAiQAEPAFAAsJAAsACwkBCwALAAsgACADNgIEIAAgAzYCACAAIAMgBEF8cWo2AgggAyEEA0AgBCIHIAEqAgA4AgAgBEEEaiEEIAFBBGoiASAGRw0ACyAAIAQ2AgQgBCADa0ECdSEFIAMgBEYNACADIQEDQCABKgIAIgkgCZQgCJIhCCABIAdHIQAgAUEEaiEBIAANAAsLAkAgCIuRIghDAAAAAF5FDQAgAyAERg0AQQEgBSAFQQFNGyIAQQFxIQRBACEBIAVBAk8EQCAAQX5xIQdBACEAA0AgAyABQQJ0IgVqIgYgBioCACAIlTgCACADIAVBBHJqIgUgBSoCACAIlTgCACABQQJqIQEgAEECaiIAIAdHDQALCyAERQ0AIAMgAUECdGoiACAAKgIAIAiVOAIACyACQSBqJAALCQBB2RYQnAEAC60RAQF/QbAPQQJByNQAQdDUAEECQQNBABAEQezUAEGM1QBBuNUAQQBByNUAQQRBy9UAQQBBy9UAQQBB7CVBzdUAQQUQBUHs1ABBAkHQ1QBB0NQAQQZBBxAGQQgQqAUiAEEANgIEIABBCDYCAEHs1ABB0SVBBEHA1gBB0NYAQQkgAEEAQQAQB0EIEKgFIgBBADYCBCAAQQo2AgBB7NQAQaQUQQJB2NYAQdDUAEELIABBAEEAEAdBhNcAQbDXAEHk1wBBAEHI1QBBDEHL1QBBAEHL1QBBAEHaJUHN1QBBDRAFQYTXAEECQfTXAEHQ1ABBDkEPEAZBCBCoBSIAQQA2AgQgAEEQNgIAQYTXAEHRJUEEQdDYAEHQ1gBBESAAQQBBABAHQQgQqAUiAEEANgIEIABBEjYCAEGE1wBBpBRBAkHg2ABB0NQAQRMgAEEAQQAQB0HA2QBBgNoAQbjaAEEAQcjVAEEUQcvVAEEAQcvVAEEAQcUWQc3VAEEVEAVBwNkAQQJByNoAQdDUAEEWQRcQBkEIEKgFIgBCgICAgBA3AwBBwNkAQcgaQQNB0NoAQdzaAEEYIABBAEEAEAdBhNsAQbDbAEHk2wBBAEHI1QBBGUHL1QBBAEHL1QBBAEHhH0HN1QBBGhAFQYTbAEEDQfTbAEHc2gBBG0EcEAZBCBCoBSIAQQA2AgQgAEEdNgIAQYTbAEGvC0EDQcjcAEHU3ABBHiAAQQBBABAHQQgQqAUiAEEANgIEIABBHzYCAEGE2wBBuCZBAkHY3QBB0NQAQSAgAEEAQQAQB0EIEKgFIgBBADYCBCAAQSE2AgBBhNsAQdALQQNB4N0AQdTcAEEiIABBAEEAEAdBCBCoBSIAQQA2AgQgAEEjNgIAQYTbAEHFC0EDQeDdAEHU3ABBIiAAQQBBABAHQQgQqAUiAEEANgIEIABBJDYCAEGE2wBByw9BBEHw3QBBgN4AQSUgAEEAQQAQB0EIEKgFIgBBADYCBCAAQSY2AgBBhNsAQb8PQQNByNwAQdTcAEEeIABBAEEAEAdBCBCoBSIAQQA2AgQgAEEnNgIAQYTbAEGhHEEFQZDeAEGk3gBBKCAAQQBBABAHQQgQqAUiAEEANgIEIABBKTYCAEGE2wBBzBNBAkGs3gBB0NQAQSogAEEAQQAQB0EIEKgFIgBBADYCBCAAQSs2AgBBhNsAQZcPQQJBrN4AQdDUAEEqIABBAEEAEAdBCBCoBSIAQQA2AgQgAEEsNgIAQYTbAEGkFEECQazeAEHQ1ABBKiAAQQBBABAHQdTeAEGA3wBBtN8AQQBByNUAQS1By9UAQQBBy9UAQQBBhC1BzdUAQS4QBUHU3gBBBEHQ3wBB4N8AQS9BMBAGQQgQqAUiAEEANgIEIABBMTYCAEHU3gBBrwtBBkHw3wBBiOAAQTIgAEEAQQAQB0EIEKgFIgBBADYCBCAAQTM2AgBB1N4AQbgmQQJB3OAAQdDUAEE0IABBAEEAEAdBCBCoBSIAQQA2AgQgAEE1NgIAQdTeAEHQC0EEQfDgAEGA3gBBNiAAQQBBABAHQQgQqAUiAEEANgIEIABBNzYCAEHU3gBBxQtBA0GA4QBB1NwAQTggAEEAQQAQB0EIEKgFIgBBADYCBCAAQTk2AgBB1N4AQbkLQQNBjOEAQdTcAEE6IABBAEEAEAdBCBCoBSIAQQA2AgQgAEE7NgIAQdTeAEGnD0EDQZjhAEHc2gBBPCAAQQBBABAHQQgQqAUiAEEANgIEIABBPTYCAEHU3gBByw9BBUGw4QBBxOEAQT4gAEEAQQAQB0EIEKgFIgBBADYCBCAAQT82AgBB1N4AQfsSQQVB0OEAQcThAEHAACAAQQBBABAHQQgQqAUiAEEANgIEIABBwQA2AgBB1N4AQcUUQQRBkOIAQeDfAEHCACAAQQBBABAHQQgQqAUiAEEANgIEIABBwwA2AgBB1N4AQd8UQQJBoOIAQdDUAEHEACAAQQBBABAHQQgQqAUiAEEANgIEIABBxQA2AgBB1N4AQc4UQQJBoOIAQdDUAEHEACAAQQBBABAHQQgQqAUiAEEANgIEIABBxgA2AgBB1N4AQcwTQQJBqOIAQdDUAEHHACAAQQBBABAHQQgQqAUiAEEANgIEIABByAA2AgBB1N4AQaIkQQNBjOEAQdTcAEE6IABBAEEAEAdBCBCoBSIAQQA2AgQgAEHJADYCAEHU3gBBtRRBA0Gw4gBB1NwAQcoAIABBAEEAEAdBCBCoBSIAQQA2AgQgAEHLADYCAEHU3gBBoCRBA0GM4QBB1NwAQTogAEEAQQAQB0EIEKgFIgBBADYCBCAAQcwANgIAQdTeAEGXD0ECQbziAEHQ1ABBzQAgAEEAQQAQB0EIEKgFIgBBADYCBCAAQc4ANgIAQdTeAEGkFEECQbziAEHQ1ABBzQAgAEEAQQAQB0EIEKgFIgBBADYCBCAAQc8ANgIAQdTeAEHVH0ECQbziAEHQ1ABBzQAgAEEAQQAQB0EIEKgFIgBBADYCBCAAQdAANgIAQdTeAEHJH0EDQYzhAEHU3ABBOiAAQQBBABAHQQgQqAUiAEEANgIEIABB0QA2AgBB1N4AQaEcQQVB0OIAQaTeAEHSACAAQQBBABAHQawpQQJB5OIAQeziAEHTAEHUAEEAEARBnOMAQdTjAEGU5ABBAEHI1QBB1QBBy9UAQQBBy9UAQQBBrhhBzdUAQdYAEAVBnOMAQQFBpOQAQcjVAEHXAEHYABAGQZzjAEHIHEECQajkAEHs4gBB2QBB2gBBABAIQZzjAEHLJkEBQbTkAEHI1QBB2wBB3ABBABAIQZzjAEHeLUEDQbjkAEHU3ABB3QBB3gBBABAIQZzjAEHyFEECQeTiAEHs4gBB0wBB3wBBABAIQZzjAEGjKUEBQbTkAEHI1QBB2wBB4ABBABAIQZzjAEG3EUECQcTkAEHQ1ABB4QBB4gBBABAIC94BAQF/IwBBIGsiAiQAIAIgATYCFAZAIAJBCGogAkEUahCZARkgAiQABkAgAigCFBAAGSACJAAQ8AUACwkACwZAIAIoAhQQABkgAiQAEPAFAAsGQCACQRRqIgEgAkEIaiAAEQAABkAgARCaASEBGSACJAAgAigCFCIABEAgAiAANgIYIAAQ8wELCQALGSACJAAgAigCCCIABEAgAiAANgIMIAAQ8wELCQALIAIoAhQiAARAIAIgADYCGCAAEPMBCyACKAIIIgAEQCACIAA2AgwgABDzAQsgAkEgaiQAIAELBgBB7NQACy4BAX8gAARAIAAoAgQhASAAQQA2AgQgAQRAIAEgASgCACgCEBECAAsgABDzAQsLKQEBfyMAQRBrIgIkACACIAE2AgwgAkEMaiAAEQEAIQAgAkEQaiQAIAALMAECfyMAIQIGQAZAQQgQqAUhARgBIAEgACgCABCdASEAGSACJAAgARDzAQkACyAAC/oDAgN/AX0jAEEwayIDJAACQAJAIAEoAgQgASgCAGtBAnUiBCAAKAIAIgVGBEAgAigCBCACKAIAa0ECdSAERg0BC0Ht4wItAAAEQCADIAU2AgBB89EAIAMQ2AELQQgQ5AUhASAAKAIAIQAgA0EBOgAvBkAgA0EEaiICIAAQ1AUgA0EBOgAuBkAgAyACQezDABDMBSIAKAIINgIYIAMgACkCADcDECAAQgA3AgAgAEEANgIIIANBAToALQZAIAMgA0EQakHTPBDOBSIAKAIINgIoIAMgACkCADcDICAAQgA3AgAgAEEANgIIIANBAToALAZAIAEgA0EgahDABSIAQYCPAjYCACADQQA6ACwgAEGkjwJB4wAQ5gUMBRkgAyQAIAMtACwhACADLAArQQBIBEAgAygCIBDzAQsgAyAAQQFxOgAtCQALABkgAyQAIAMtAC0hACADLAAbQQBIBEAgAygCEBDzAQsgAyAAQQFxOgAuCQALABkgAyQAIAMtAC4hACADLAAPQQBIBEAgAygCBBDzAQsgAyAAQQFxOgAvCQALABkgAyQAIAMtAC8EQCABEOUFCwkACwALIAAoAgQiBCAEKAIAKAIEEQEAIQQgASgCACACKAIAIAAoAgQiACAAKAIAKAIIEQEAIAQRCwAhBiADQTBqJAAgBg8LAAvKAgICfwF9IwBBIGsiBCQAIAEgACgCBCIFQQF1aiEBIAAoAgAhACAFQQFxBEAgASgCACAAaigCACEACyAEIAI2AgQGQCAEQRBqIARBBGoQmQEZIAQkAAZAIAQoAgQQABkgBCQAEPAFAAsJAAsGQCAEKAIEEAAZIAQkABDwBQALIAQgAzYCHAZABkAgBEEEaiAEQRxqEJkBGSAEJAAGQCAEKAIcEAAZIAQkABDwBQALCQALBkAgBCgCHBAAGSAEJAAQ8AUACwZAIAEgBEEQaiAEQQRqIAARCwAhBhkgBCQAIAQoAgQiAARAIAQgADYCCCAAEPMBCwkACxkgBCQAIAQoAhAiAARAIAQgADYCFCAAEPMBCwkACyAEKAIEIgAEQCAEIAA2AgggABDzAQsgBCgCECIABEAgBCAANgIUIAAQ8wELIARBIGokACAGCwcAIAAoAgALNQEBfyABIAAoAgQiAkEBdWohASAAKAIAIQAgASACQQFxBH8gASgCACAAaigCAAUgAAsRAQALBgBBhNcAC2wBAn8jACEBQQgQqAUhAiAAKAIAIQAgAkEANgIEIAIgADYCAAZAQRAQqAUhARkgASQAIAJBADYCBCACEPMBCQALIAEgADYCDCABQeQANgIEIAFBhNgANgIAIAEgAEECdDYCCCACIAE2AgQgAguXAwIDfwF9IwBBIGsiAyQAAkACQCABKAIEIAEoAgBrQQJ1IgQgACgCACIFRgRAIAIoAgQgAigCAGtBAnUgBEYNAQtB7eMCLQAABEAgAyAFNgIAQfPRACADENgBC0EIEOQFIQEgACgCACEAIANBAToAHwZAIANBBGoiAiAAENQFIANBAToAHgZAIAMgAkHswwAQzAUiACgCCDYCGCADIAApAgA3AxAgAEIANwIAIABBADYCCCADQQE6AB0GQCABIANBEGoQwAUiAEGAjwI2AgAgA0EAOgAdIABBpI8CQeMAEOYFDAQZIAMkACADLQAdIQAgAywAG0EASARAIAMoAhAQ8wELIAMgAEEBcToAHgkACwAZIAMkACADLQAeIQAgAywAD0EASARAIAMoAgQQ8wELIAMgAEEBcToAHwkACwAZIAMkACADLQAfBEAgARDlBQsJAAsACyAAKAIEIgQgBCgCACgCBBEBACEEIAEoAgAgAigCACAAKAIEIgAgACgCACgCCBEBACAEEQsAIQYgA0EgaiQAIAYPCwALDQAgACgCAEEEaygCAAsuAQF/IAAEQCMAIQEgAEHw2AA2AgAGQCAAKAIEEAAZIAEkABDwBQALIAAQ8wELC1kBAX8jAEEQayICJAAgAiABNgIMBkAgAkEMaiAAEQEAIQAZIAIkAAZAIAIoAgwQABkgAiQAEPAFAAsJAAsGQCACKAIMEAAZIAIkABDwBQALIAJBEGokACAAC4IBAQN/IwBBEGsiASQAQQgQqAUhAiAAKAIAIQMgAEEANgIAIAJB8NgANgIABkAgAxACIAEgAzYCCEGY1AAgAUEIahAJIQAZIAEkAAZAIAMQABkgASQAEPAFAAsgAhDzAQkACyACIAA2AgQGQCADEAAZIAEkABDwBQALIAFBEGokACACCzcBAX8gASAAKAIEIgNBAXVqIQEgACgCACEAIAEgAiADQQFxBH8gASgCACAAaigCAAUgAAsRAwALBgBBhNsACzwBAX8gAARAIAAoAggiAQRAIAEgASgCACgCEBECAAsgACgCBCIBBEAgASABKAIAKAIUEQIACyAAEPMBCwvjAQEEfyMAQRBrIgMkAAJAIAEoAgAiBEHw////B0kEQAJAAkAgBEELTwRAIARBD3JBAWoiBRCoBSEGIAMgBUGAgICAeHI2AgwgAyAGNgIEIAMgBDYCCCAEIAZqIQUMAQsgAyAEOgAPIANBBGoiBiAEaiEFIARFDQELIAYgAUEEaiAEENIBGgsgBUEAOgAAIAMgAjYCAAZAIANBBGogAyAAEQMAIQAMAhkgAyQAIAMsAA9BAEgEQCADKAIEEPMBCwkACwALEDMACyADLAAPQQBIBEAgAygCBBDzAQsgA0EQaiQAIAAL9AMBBX8jACEFBkAGQEEQEKgFIQQYAQJ/IAEoAgAhAiMAQSBrIgEkACAEQgA3AgQgBCACNgIAIARBADoADAJAAkACQAJAIAAoAgQgAC0ACyIDIAPAIgZBAEgiAxtBAmsOBQADAwMBAwsgACgCACAAIAMbIgMvAABB7OQARgRAQRAQqAUiACACNgIMIABB5gA2AgQgAEHg1QA2AgAgACACQQJ0NgIIDAILIAMvAABB6eABRw0CQRAQqAUiACACNgIMIABB5AA2AgQgAEGE2AA2AgAgACACQQJ0NgIIDAELIAAoAgAgACADG0HBJEEGENcBDQFBEBCoBSIAIAI2AgwgAEHkADYCBCAAQYTYADYCACAAIAJBAnQ2AgggBEEBOgAMCyAEIAA2AgggAUEgaiQAIAQMAQtB7eMCLQAABEAgASAAKAIAIAAgBkEASBs2AgBBsNAAIAEQ2AELQQgQ5AUhAiABQQE6AB8GQCABQRBqIgMgABDTBSABQQE6AB4GQCACIAMQwAUiAEGAjwI2AgAgAUEAOgAeIABBpI8CQeMAEOYFGSABJAAgAS0AHiEAIAEsABtBAEgEQCABKAIQEPMBCyABIABBAXE6AB8JAAsZIAEkACABLQAfBEAgAhDlBQsJAAsACyEAGSAFJAAgBBDzAQkACyAAC08BAn8jACEDIAAoAgQiAgRAIAIgAigCACgCFBECAAsGQAZAQcwAEKgFIQIYASACIAAoAgggARCIASEBGSADJAAgAhDzAQkACyAAIAE2AgQLNwEBfyABIAAoAgQiA0EBdWohASAAKAIAIQAgASACIANBAXEEfyABKAIAIABqKAIABSAACxEAAAtLAQF/IwBBEGsiAiQAIAACfyABKAIERQRAIAJBADYCCEGAigIgAkEIahAJDAELIAJBATYCCEGAigIgAkEIahAJCzYCACACQRBqJAALhQEBAn8jAEEQayICJAAgASAAKAIEIgNBAXVqIQEgACgCACEAIAJBDGogASADQQFxBH8gASgCACAAaigCAAUgAAsRAAAGQCACKAIMEAIZIAIkAAZAIAIoAgwQABkgAiQAEPAFAAsJAAsGQCACKAIMIgAQABkgAiQAEPAFAAsgAkEQaiQAIAALnQgBBH8jAEHQAGsiAiQAIAAoAgQiAwRAIAMgAygCACgCFBECAAsGQEHMABCoBSEDIAAoAgghBCADQgA3AgQgA0Hk3AA2AgAgA0IANwIMIANBADYCFCADQgA3AhwgA0IANwIkIANCADcCLCADQgA3AjQgA0IANwI8IANCgICAgICAgMA/NwJEBkAgAyABIAQQiQEZIAIkACADQThqEIoBIAMQ8wEJAAsHACEBIAIkAEHkhgNBkAg2AgBB4IYDQQA2AgAgARCVBgJAQeiGAygCACIDQQNGBEAGQAZAIAJBOGohAyABEOkFIgEgASgCACgCCBEBACEBGAQgAyABEDIhAQZAIAJBLGpByygQMiEDBkAgASADEIsBQX9HBEAgAiAAKAIEKAIINgIAQZ7NACACENgBBkBBCBDkBSEEGAcgACgCBCgCCCEAIAJBAToATwZAIAJBFGoiBSAAENQFIAJBAToATgZAIAIgBUHvyQAQzAUiACgCCDYCKCACIAApAgA3AyAgAEIANwIAIABBADYCCCACQQE6AE0GQCAEIAJBIGoQwwUhACACQQA6AE0gAEGskAJBARDmBQwIGSACJAAgAi0ATSEAIAIsACtBAEgEQCACKAIgEPMBCyACIABBAXE6AE4JAAsAGSACJAAgAi0ATiEAIAIsAB9BAEgEQCACKAIUEPMBCyACIABBAXE6AE8JAAsAGSACJAAgAi0ATwRABkAgBBDlBRgJCwkACwALEOsFDAQZIAIkACADLAALQQBIBEAgAygCABDzAQsJAAsAGSACJAAgASwAC0EASARAIAEoAgAQ8wELCQALABkgAiQABkAQ6gUZIAIkABDwBQALCQALAAsgARDpBSEAIANBAkYEQCACIAAgACgCACgCCBEBADYCEEH8zgAgAkEQahDYAUEIEOQFIQMgACAAKAIAKAIIEQEAIQAgAkEBOgBMBkAgAkEsaiAAEDIhACACQQE6AEsGQCACQUBrIABB/ccAEMwFIgEoAgg2AgAgAiABKQIANwM4IAFCADcCACABQQA2AgggAkEBOgBKBkAgAyACQThqEMMFIQEgAkEAOgBKIAFBrJACQQEQ5gUMBBkgAiQAIAItAEohASACLABDQQBIBEAgAigCOBDzAQsgAiABQQFxOgBLCQALABkgAiQAIAItAEshASAALAALQQBIBEAgACgCABDzAQsgAiABQQFxOgBMCQALABkgAiQAIAItAEwEQCADEOUFCwZAEOoFGSACJAAQ8AUACwkACwALQcQ4EOABBkAGQAZAQQgQ5AUhABgEIABBxDgQxAUhABkgAiQABkAgABDlBRgECQALIABBrJACQQEQ5gUZIAIkAAZAEOoFGSACJAAQ8AUACwkACwsACyAAIAM2AgQgAkHQAGokAAuDAgEEfyMAQRBrIgMkACABIAAoAgQiBEEBdWohBiAAKAIAIQEgBEEBcQRAIAYoAgAgAWooAgAhAQsCQCACKAIAIgBB8P///wdJBEACQAJAIABBC08EQCAAQQ9yQQFqIgUQqAUhBCADIAVBgICAgHhyNgIMIAMgBDYCBCADIAA2AgggACAEaiEFDAELIAMgADoADyADQQRqIgQgAGohBSAARQ0BCyAEIAJBBGogABDSARoLIAVBADoAAAZAIAYgA0EEaiABEQAADAIZIAMkACADLAAPQQBIBEAgAygCBBDzAQsJAAsACxAzAAsgAywAD0EASARAIAMoAgQQ8wELIANBEGokAAumAQEBfyMAQRBrIgIkAAJAIAAoAgQiAEUEQAZABkBBCBDkBSEAGAMgAEH+ORDEBSEADAIZIAIkACAAEOUFCQALAAsgACABIAAoAgAoAgwRAAAgAkEBNgIMBkBBACACQQxqEIMBGSACJAAGQCACKAIMEAAZIAIkABDwBQALCQALBkAgAigCDBAAGSACJAAQ8AUACyACQRBqJAAPCyAAQayQAkEBEOYFAAv8CQIIfwJ9IwBBQGoiAyQAAkAgACgCBCIHRQRABkAGQEEIEOQFIQAY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BkAgA0EgaiABIAIQlgEGQCAAEAo2AgBBACEBIANBADYCECADKAIkIgIgAygCICIERwRAIAQhAgNABkAgAiABQQJ0aiEEIwBBEGsiASQAIAAoAgAhBSABIAMoAhA2AghB7IoCIAFBCGoiBhAJIQIgASAEKgIAOAIIBkBBkIsCIAYQCSEEBkAgBSACIAQQDRkgASQABkAgBBAAGSABJAAQ8AUACwkACxkgASQABkAgAhAAGSABJAAQ8AUACwkACwZAIAQQABkgASQAEPAFAAsGQCACEAAZIAEkABDwBQALIAFBEGokABkgAyQABkAgACgCABAAGSADJAAQ8AUACwkACyADIAMoAhBBAWoiATYCECABIAMoAiQgAygCICICa0ECdUkNAAsLIAIEQCADIAI2AiQgAhDzAQsgA0EwaiQADxkgAyQAIAMoAiAiAARAIAMgADYCJCAAEPMBCwkACwAHACEAIAMkAEHkhgNBzAg2AgBB4IYDQQA2AgAgABCVBgJAQeiGAygCAEEBRgRAIAAQ6QUhAEEIEOQFIQIgACAAKAIAKAIIEQEAIQAgA0EBOgAvBkAgA0EEaiAAEDIhACADQQE6AC4GQCADIABB+8sAEMwFIgEoAgg2AhggAyABKQIANwMQIAFCADcCACABQQA2AgggA0EBOgAtBkAgAiADQRBqEMMFIQEgA0EAOgAtIAFBrJACQQEQ5gUMBBkgAyQAIAMtAC0hASADLAAbQQBIBEAgAygCEBDzAQsgAyABQQFxOgAuCQALABkgAyQAIAMtAC4hASAALAALQQBIBEAgACgCABDzAQsgAyABQQFxOgAvCQALABkgAyQAIAMtAC8EQCACEOUFCwZAEOoFGSADJAAQ8AUACwkACwALCQELAAsACyAAQayQAkEBEOYFAAuHAQECfyMAQRBrIgMkACABIAAoAgQiBEEBdWohASAAKAIAIQAgA0EMaiABIAIgBEEBcQR/IAEoAgAgAGooAgAFIAALEQUABkAgAygCDBACGSADJAAGQCADKAIMEAAZIAMkABDwBQALCQALBkAgAygCDCIAEAAZIAMkABDwBQALIANBEGokACAAC7ELAgl/An0jAEHgAGsiBCQABkACQAJAIAAoAgQiC0UEQEHt4wItAAAEQEH+ORDgAQsGQAZAQQgQ5AUhABgFIABB/jkQxAUhAAwCGSAE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ABCVBgJAQeiGAygCAEEBRgRAIAAQ6QUhAEHt4wItAAAEQAZAIAAgACgCACgCCBEBACEBGAcgAiABNgIAQcfOACACENgBCwZAQQgQ5AUhAyAAIAAoAgAoAggRAQAhABgGIAJBAToAHwZAIAJBBGogABAyIQAgAkEBOgAeBkAgAiAAQb3DABDMBSIBKAIINgIYIAIgASkCADcDECABQgA3AgAgAUEANgIIIAJBAToAHQZAIAMgAkEQahDDBSEBIAJBADoAHSABQayQAkEBEOYFDAQZIAIkACACLQAdIQEgAiwAG0EASARAIAIoAhAQ8wELIAIgAUEBcToAHgkACwAZIAIkACACLQAeIQEgACwAC0EASARAIAAoAgAQ8wELIAIgAUEBcToAHwkACwAZIAIkACACLQAfBEAGQCADEOUFGAgLBkAQ6gUZIAIkABDwBQALCQALAAsJAQsACyABQQRqIQEMAAsACyAAQayQAkEBEOYFGSACJAAJAAsAC78BAQJ/IwBBEGsiAyQAIAEgACgCBCIEQQF1aiEBIAAoAgAhACAEQQFxBEAgASgCACAAaigCACEACyADIAI2AgwGQCADIANBDGoQywEZIAMkAAZAIAMoAgwQABkgAyQAEPAFAAsJAAsGQCADKAIMEAAZIAMkABDwBQALBkAgASADIAARAAAZIAMkACADKAIAIgAEQCADIAA2AgQgABDzAQsJAAsgAygCACIABEAgAyAANgIEIAAQ8wELIANBEGokAAt5AQJ/IwAhAgZAAkAgACgCBCIDRQRAQe3jAi0AAARAQf45EOABCwZABkBBCBDkBSEAGAQgAEH+ORDEBSEADAIZIAIkAAZAIAAQ5QUYBAkACwALIAMgARCYASAAEJUBIAAQkwEPCyAAQayQAkEBEOYFGSACJAAJAAsAC18BAX8jACEBAkAgACgCBCIARQRAQe3jAi0AAARAQf45EOABCwZABkBBCBDkBSEAGAMgAEH+ORDEBSEADAIZIAEkACAAEOUFCQALAAsgACgCCA8LIABBrJACQQEQ5gUAC18BAX8jACEBAkAgACgCBCIARQRAQe3jAi0AAARAQf45EOABCwZABkBBCBDkBSEAGAMgAEH+ORDEBSEADAIZIAEkACAAEOUFCQALAAsgACgCKA8LIABBrJACQQEQ5gUAC2EBAX8jACECAkAgACgCBCIARQRAQe3jAi0AAARAQf45EOABCwZABkBBCBDkBSEAGAMgAEH+ORDEBSEADAIZIAIkACAAEOUFCQALAAsgACABNgIoDwsgAEGskAJBARDmBQALqxECB38CfSMAQYABayIFJAACQCABKAIEIgZFBEBB7eMCLQAABEBB/jkQ4AELBkAGQEEIEOQFIQAYAyAAQf45EMQFIQAMAhkgBSQAIAAQ5QUJAAsACwJAIAIoAgQgAigCAGtBAnUiByABKAIAIghHBEBB7eMCLQAABEAgBSAHNgIUIAUgCDYCEEHY0gAgBUEQahDYAQtBCBDkBSEDIAEoAgAhACAFQQE6AHsGQCAFQTxqIgEgABDUBSAFQQE6AHoGQCAFIAFBwccAEMwFIgAoAgg2AlAgBSAAKQIANwNIIABCADcCACAAQQA2AgggBUEBOgB5BkAgBSAFQcgAakGRwwAQzgUiACgCCDYCYCAFIAApAgA3A1ggAEIANwIAIABBADYCCCACKAIAIQAgAigCBCEBIAVBAToAeAZAIAVBMGoiAiABIABrQQJ1ENQFIAVBAToAdwZAIAUgBUHYAGogBSgCMCACIAUtADsiAMBBAEgiARsgBSgCNCAAIAEbEMsFIgAoAgg2AnAgBSAAKQIANwNoIABCADcCACAAQQA2AgggBUEBOgB2BkAgBSAFQegAakHSPBDOBSIAKAIINgIoIAUgACkCADcDICAAQgA3AgAgAEEANgIIIAVBAToAdQZAIAMgBUEgahDABSIAQYCPAjYCACAFQQA6AHUgAEGkjwJB4wAQ5gUMCBkgBSQAIAUtAHUhACAFLAArQQBIBEAgBSgCIBDzAQsgBSAAQQFxOgB2CQALABkgBSQAIAUtAHYhACAFLABzQQBIBEAgBSgCaBDzAQsgBSAAQQFxOgB3CQALABkgBSQAIAUtAHchACAFLAA7QQBIBEAgBSgCMBDzAQsgBSAAQQFxOgB4CQALABkgBSQAIAUtAHghACAFLABjQQBIBEAgBSgCWBDzAQsgBSAAQQFxOgB5CQALABkgBSQAIAUtAHkhACAFLABTQQBIBEAgBSgCSBDzAQsgBSAAQQFxOgB6CQALABkgBSQAIAUtAHohACAFLABHQQBIBEAgBSgCPBDzAQsgBSAAQQFxOgB7CQALABkgBSQAIAUtAHsEQCADEOUFCwkACwALIAMgBigCBCIGSwRAQe3jAi0AAARAIAUgBjYCAEGV0wAgBRDYAQtBCBDkBSECIAEoAgQoAgQhACAFQQE6AH8GQCAFQdgAaiIBIAAQ1AUgBUEBOgB+BkAgBSABQZvLABDMBSIAKAIINgJwIAUgACkCADcDaCAAQgA3AgAgAEEANgIIIAVBAToAfQZAIAUgBUHoAGpB0jwQzgUiACgCCDYCKCAFIAApAgA3AyAgAEIANwIAIABBADYCCCAFQQE6AHwGQCACIAVBIGoQwAUiAEGAjwI2AgAgBUEAOgB8IABBpI8CQeMAEOYFDAUZIAUkACAFLQB8IQAgBSwAK0EASARAIAUoAiAQ8wELIAUgAEEBcToAfQkACwAZIAUkACAFLQB9IQAgBSwAc0EASARAIAUoAmgQ8wELIAUgAEEBcToAfgkACwAZIAUkACAFLQB+IQAgBSwAY0EASARAIAUoAlgQ8wELIAUgAEEBcToAfwkACwAZIAUkACAFLQB/BEAgAhDlBQsJAAsACwJAIANFBEBB7eMCLQAABEBBjjwQ4AELBkAGQEEIEOQFIQAYBSAAQY48EI4BIQAMAhkgBSQAIAAQ5QUJAAsAC0EAIQYgBCgCACIEQQFrQQJPBEBBCBCoBSEGBkAgBBACIAZB8NgANgIABkAgBBACIAUgBDYCIEGY1AAgBUEgahAJIQcZIAUkAAZAIAQQABkgBSQAEPAFAAsJAAsZIAUkACAGEPMBCQALIAYgBzYCBAZAIAQQABkgBSQAEPAFAAsLAkAgAS0AWEUEQCACKAIAIQQMAQsgAigCBCIHIAIoAgAiBEYiCEUEQCAEIQIDQCACKgIAIg0gDZQgDJIhDCACQQRqIgIgB0cNAAsLIAyLkSIMQwAAAABeRQ0AIAgNAEEBIAcgBGsiB0ECdSICIAJBAU0bIghBAXEhCkEAIQIgB0EITwRAIAhBfnEhCEEAIQcDQCAEIAJBAnQiCWoiCyALKgIAIAyVOAIAIAQgCUEEcmoiCSAJKgIAIAyVOAIAIAJBAmohAiAHQQJqIgcgCEcNAAsLIApFDQAgBCACQQJ0aiICIAIqAgAgDJU4AgALIAVBIGogASgCBCIBIAQgAyAGIAEoAgAoAgQRCgAgBSgCICEBIAUoAiQhAgZAIAUQCjYCWAZAEAohAyACIAFrQQN1IQIgBSADNgJIIAVB6ABqQQRyIQEGQAJAA0ACQCAFIAJBAWs2AjwgAkEATARAIAZFDQMgBkHw2AA2AgAGQCAGKAIEEAAMAhkgBSQAEPAFAAsACyAFIAUoAiApAgA3A2ggBUHYAGogBUE8aiICIAVB6ABqEI8BIAVByABqIAIgARCQASAFKAIgIgIgBSgCJCIDIAMgAmtBA3UQkQEgBSAFKAIkQQhrNgIkIAUoAjwhAgwBCwsgBhDzAQsgABALIgA2AgAGQEH/FBAMIQEGQCAAIAEgBSgCWBANGSAFJAAGQCABEAAZIAUkABDwBQALCQALBkAgARAAGSAFJAAQ8AUAC0HbExAMIQEGQCAAIAEgBSgCSBANGSAFJAAGQCABEAAZIAUkABDwBQALCQALGSAFJAAGQCAAEAAZIAUkABDwBQALCQALGSAFJAAGQCAFKAJIEAAZIAUkABDwBQALCQALGSAFJAAGQCAFKAJYEAAZIAUkABDwBQALCQALGSAFJAAgBSgCICIABEAgBSAANgIkIAAQ8wELCQALBkAgARAAGSAFJAAQ8AUACwZAIAUoAkgQABkgBSQAEPAFAAsGQCAFKAJYEAAZIAUkABDwBQALIAUoAiAiAARAIAUgADYCJCAAEPMBCyAFQYABaiQADwsgAEGkjwJB4wAQ5gUACwALIABBrJACQQEQ5gUACwkAIAEgABECAAsGAEGc4wALDAAgAARAIAAQ8wELCwcAIAARDwALBwBBARCoBQvWAQEEfyMAQRBrIgIkAAJAIAEoAgAiA0Hw////B0kEQAJAAkAgA0ELTwRAIANBD3JBAWoiBBCoBSEFIAIgBEGAgICAeHI2AgwgAiAFNgIEIAIgAzYCCCADIAVqIQQMAQsgAiADOgAPIAJBBGoiBSADaiEEIANFDQELIAUgAUEEaiADENIBGgsgBEEAOgAABkAgAkEEaiAAEQIADAIZIAIkACACLAAPQQBIBEAgAigCBBDzAQsJAAsACxAzAAsgAiwAD0EASARAIAIoAgQQ8wELIAJBEGokAAu8AQEBfyMAQRBrIgEkAEHs4wItAABFBEAGQEHw4wJBoAtBDhDKBSABIAAoAgAgACAALAALQQBIGzYCACABQfDjAkHw4wIoAgBB++MCLAAAQQBOGzYCBEG33wJBsOQAIAEQDhpB7OMCQQE6AAAgAUEBNgIMBkBBASABQQxqEIMBGSABJAAGQCABKAIMEAAZIAEkABDwBQALCQALGSABJAAJAAsGQCABKAIMEAAZIAEkABDwBQALCyABQRBqJAALCQBB7OMCLQAAC1cBAX8jAEEQayIDJAAgAyACNgIMBkAgASADQQxqIAARAAAZIAMkAAZAIAMoAgwQABkgAyQAEPAFAAsJAAsGQCADKAIMEAAZIAMkABDwBQALIANBEGokAAu5AQEBfyMAIQJB7eMCLQAABEBB+zsQ4AELAkBB7OMCLQAARQRAQe3jAi0AAARAQZ8tEOABCwZABkBBCBDkBSEAGAMgAEGfLRDEBSEADAIZIAIkACAAEOUFCQALAAtB5OMCLQAABEBB4OMCKAIAGgtB5OMCQQE6AABB4OMCQcjjAjYCACABKAIAEAJB6OMCKAIAEABB6OMCIAEoAgA2AgBB7uMCQQA6AAAgABAPDwsgAEGskAJBARDmBQALCwBB7eMCIAA6AAALCQBB7uMCLQAAC9oBAQR/IwBBEGsiAiQAAkAgASgCACIDQfD///8HSQRAAkACQCADQQtPBEAgA0EPckEBaiIEEKgFIQUgAiAEQYCAgIB4cjYCDCACIAU2AgQgAiADNgIIIAMgBWohBAwBCyACIAM6AA8gAkEEaiIFIANqIQQgA0UNAQsgBSABQQRqIAMQ0gEaCyAEQQA6AAAGQCACQQRqIAARAQAhAAwCGSACJAAgAiwAD0EASARAIAIoAgQQ8wELCQALAAsQMwALIAIsAA9BAEgEQCACKAIEEPMBCyACQRBqJAAgAAvTDAIKfwF+IwBBMGsiAiQAAn8GQEH04wIoAgBB++MCLAAAIgVB/wFxIAVBAEgbIgRBAWoiAUHw////B08EQBAzAAsCQCABQQpNBEAgAkEANgIQIAJCADcDCCACIAE6ABMgAkEIaiEDDAELIAFBD3JBAWoiBhCoBSEDIAIgATYCDCACIAM2AgggAiAGQYCAgIB4cjYCEAsgBARAIANB8OMCQfDjAigCACAFQQBOGyAEENMBGgsgAyAEakEvOwAABkAgAkEIaiAAKAIAIAAgAC0ACyIBwEEASCIDGyAAKAIEIAEgAxsQywUhABkgAiQAIAIsABNBAEgEQCACKAIIEPMBCwkACyACIAAoAgg2AiAgAiAAKQIANwMYIABCADcCACAAQQA2AgggAiwAE0EASARAIAIoAggQ8wELIAJBADYCECACQgA3AwgGQAZAIAJBCGohBCACKAIYIAJBGGogAiwAIyIBQQBIIgMbIQAgACACKAIcIAFB/wFxIAMbaiEIIwBBEGsiAyQAAkAgACAIRg0AIAQoAgghASAEKAIEIAQtAAsiBSAFwEEASCIHGyEGIAggAGshBQJAAkACQCAAIAQoAgAiCSAEIAcbIgpPIAYgCmogAE9xRQRAIAUgAUH/////B3FBAWtBCiAHGyIHIAZrTQR/IAFBGHYFIAQgByAFIAZqIAdrIAYgBhDIBSAEKAIAIQkgBC0ACwshASAAQX9zIAhqIQpBACEHIAkgBCABwEEASBsgBmohASAFQQdxIgkEQANAIAEgAC0AADoAACAAQQFqIQAgAUEBaiEBIAdBAWoiByAJRw0ACwsgCkEHTwRAA0AgASAALQAAOgAAIAEgAC0AAToAASABIAAtAAI6AAIgASAALQADOgADIAEgAC0ABDoABCABIAAtAAU6AAUgASAALQAGOgAGIAEgAC0ABzoAByABQQhqIQEgAEEIaiIAIAhHDQALCyABQQA6AAAgBSAGaiEAIAQsAAtBAE4NASAEIAA2AgQMBAsgBUHw////B08NAQJAIAVBCk0EQCADIAU6AA8gA0EEaiEBDAELIAVBD3JBAWoiBhCoBSEBIAMgBkGAgICAeHI2AgwgAyABNgIEIAMgBTYCCAsgASAAIAUQ0gEgBWpBADoAAAZAIAQgAygCBCADQQRqIAMtAA8iAMBBAEgiARsgAygCCCAAIAEbEMsFGgwDGSADJAAgAywAD0EASARAIAMoAgQQ8wELCQALAAsgBCAAOgALDAILEDMACyADLAAPQQBODQAgAygCBBDzAQsgA0EQaiQAGSACJAAgAiwAE0EASARAIAIoAggQ8wELCQALBkAjAEHwAGsiBSQAIwBBEGsiAyQAIANBCGoiAEEANgIAQbmCAy0AAEUEQEG5ggNBAToAAAsgAEHA2wI2AgQCfyACQQhqIgQiAS0AC0EHdgRAIAEoAgAMAQsgAQsgBRApIgFBgWBPBH9BuOwCQQAgAWs2AgBBfwUgAQtBf0YEQEG47AIoAgAhAUG4ggMtAABFBEBBuIIDQQE6AAALIANBvNsCNgIEIAMgATYCACAAIAMpAwA3AwALIwBBEGsiASQAAkACQCAAKAIARQ0AIAAoAgBBLEcEQCAAKAIAQTZHDQELIAJB//8DNgIsIAJB/wE6ACgMAQsgACgCAARAIAFBADYCDCABIAQ2AgggAUEANgIEIAFB2BA2AgAjAEEQayIEJAAgBEEANgIMBkAgASAAELYFBwAhACAEJAAgABDpBRoGQBDrBRkgBCQABkAQ6gUZIAQkABDwBQALCQALAAsgBEEQaiQAIAJB//8DNgIsIAJBADoAKAwBCyACQf//AzYCLCACQQA6ACggAiAFKAIMQYDgA3FBgCBrIgBB//8CTQR/IABBDHZB9PABai0AAAVBCAvAOgAoIAIgBSgCDEH/H3E2AiwLIAFBEGokACADQRBqJAAgBUHwAGokABkgAiQAIAIsABNBAEgEQCACKAIIEPMBCwkACxkgAiQAIAIsACNBAEgEQCACKAIYEPMBCwkACwcAIQAgAiQAIAAQ6QUaEOoFQQAMAQsgAjEAKCELIAIsABNBAEgEQCACKAIIEPMBCyACLAAjQQBIBEAgAigCGBDzAQsgC0L/AVIgC0IAUnELIQAgAkEwaiQAIAAL+wEBAn8jACEDIABCADcCICAAIAI2AgggAEHk3AA2AgAgAEIANwIoIABCADcCMCAAQThqIgRCADcCACAAQUBrQgA3AgAgAEGAgID8AzYCSAZAIAAgASABKAIAKAIAEQEANgIUIAAgASABKAIAKAIEEQEANgIYIAAgASABKAIAKAIIEQEANgIcIAAgACgCFEEEaiIBNgIQIAAgASACbBDyASIBNgIEAkAgAUUEQAZABkBBCBDkBSEAGAQgAEGPKxDEBSEADAIZIAMkAAZAIAAQ5QUYBAkACwALIABBADYCDCAADwsgAEGskAJBARDmBRkgAyQAIAQQigEJAAsAC9gCAQF/IwBBwAFrIgMkAAZABkAgA0EEaiABELABIQEYASABIABBCGpBBBCkAiABIABBEGpBBBCkAiABIABBDGpBBBCkAiAAIAIgAigCACgCABEBADYCFCAAIAIgAigCACgCBBEBADYCGCAAIAIgAigCACgCCBEBADYCHCAAIAAoAhRBBGoiAjYCECAAIAAoAgggAmwiAhDyASIANgIEAkAgAEUEQAZABkBBCBDkBSEAGAQgAEHaKhDEBSEADAIZIAMkAAZAIAAQ5QUYBAkACwALIAEgACACEKQCIAFBCGoiABC9AkUEQCABIAEoAgBBDGsoAgBqIgIgAigCEEEEchDeAgsgAUGslgEoAgAiAjYCACABIAJBDGsoAgBqQbiWASgCADYCACAAELwCGiABQewAahCKAiADQcABaiQADwsgAEGskAJBARDmBRkgAyQAIAEQsQEaCQALAAs7AQJ/IAAoAggiAgRAA0AgAigCACEBIAIQ8wEgASICDQALCyAAKAIAIQEgAEEANgIAIAEEQCABEPMBCwvCAQEFfyAAKAIEIAAtAAsiAiACwEEASCICGyIDQQBPBH8gASgCBCABLQALIgQgBMBBAEgiBhsiBUUEQEEADwsCQAJAIAMgACgCACAAIAIbIgNqIgIgAyIEayIAIAVIDQAgASgCACABIAYbIgEtAAAhBgNAIAAgBWtBAWoiAEUNASAEIAYgABDWASIARQ0BIAAgASAFENcBRQ0CIAIgAEEBaiIEayIAIAVODQALCyACIQALQX8gACADayAAIAJGGwVBfwsLpAYCBX8CfSMAIQggAigCACEGIAACfwJAIAEoAgQiBEUNAAJAIARpIgdBAk8EQCAGIQUgBCAGTQRAIAYgBHAhBQsgASgCACAFQQJ0aigCACICRQ0CIAdBAU0NAQNAIAIoAgAiAkUNAyAGIAIoAgQiB0cEQCAEIAdNBH8gByAEcAUgBwsgBUcNBAsgAigCCCAGRw0AC0EADAMLIAEoAgAgBEEBayAGcSIFQQJ0aigCACICRQ0BCyAEQQFrIQcDQCACKAIAIgJFDQEgBiACKAIEIglHIAcgCXEgBUdxDQEgAigCCCAGRw0AC0EADAELQRAQqAUhAiADKAIAKAIAIQMgAkEANgIMIAIgAzYCCCACIAY2AgQgAkEANgIAAkBBACAEIAEoAgxBAWqzIgogASoCECILIASzlF4bDQBBAiEFBkACQAJAIAQgBEEBa3FBAEcgBEEDSXIgBEEBdHIiAwJ/IAogC5WNIgpDAACAT10gCkMAAAAAYHEEQCAKqQwBC0EACyIHIAMgB0sbIgNBAUYNACADIANBAWtxRQRAIAMhBQwBCyADEPoBIQUgASgCBCEECyAEIAVPBEAgBCAFTQ0BIARBA0khBwJ/IAEoAgyzIAEqAhCVjSIKQwAAgE9dIApDAAAAAGBxBEAgCqkMAQtBAAshAyAFAn8CQCAHDQAgBGlBAUsNACADQQFBICADQQFrZ2t0IANBAkkbDAELIAMQ+gELIgMgAyAFSRsiBSAETw0BCyABIAUQrwELGSAIJAAgAhDzAQkACyABKAIEIgQgBEEBayIDcUUEQCADIAZxIQUMAQsgBCAGSwRAIAYhBQwBCyAGIARwIQULAkACQCABKAIAIAVBAnRqIgUoAgAiA0UEQCACIAFBCGoiAygCADYCACABIAI2AgggBSADNgIAIAIoAgAiA0UNAiADKAIEIQMCQCAEIARBAWsiBXFFBEAgAyAFcSEDDAELIAMgBEkNACADIARwIQMLIAEoAgAgA0ECdGohAwwBCyACIAMoAgA2AgALIAMgAjYCAAsgASABKAIMQQFqNgIMQQELOgAEIAAgAjYCAAvuBAEIfyMAQRBrIgYkAAJAIAAoAgQiA0UNACAAKAIAAn8gASgCACIEIANBAWtxIANpIgJBAU0NABogBCADIARLDQAaIAQgA3ALIgVBAnRqKAIAIgFFDQAgASgCACIBRQ0AAkAgAkEBTQRAIANBAWshAwNAAkAgBCABKAIEIgJHBEAgAiADcSAFRg0BDAULIAEoAgggBEYNAwsgASgCACIBDQALDAILA0ACQCAEIAEoAgQiAkcEQCACIANPBH8gAiADcAUgAgsgBUYNAQwECyABKAIIIARGDQILIAEoAgAiAQ0ACwwBCyABKAIEIQUCQCAAIgMoAgQiAmkiCEEBTQRAIAJBAWsgBXEhBQwBCyACIAVLDQAgBSACcCEFCyADKAIAIAVBAnRqIgcoAgAhAANAIAAiBCgCACIAIAFHDQALAkAgA0EIaiIJIARHBEAgBCgCBCEAAkAgCEEBTQRAIAAgAkEBa3EhAAwBCyAAIAJJDQAgACACcCEACyAAIAVGDQELIAEoAgAiAARAIAAoAgQhAAJAIAhBAU0EQCAAIAJBAWtxIQAMAQsgACACSQ0AIAAgAnAhAAsgACAFRg0BCyAHQQA2AgALIAQCf0EAIAEoAgAiB0UNABogBygCBCEAAkAgCEEBTQRAIAAgAkEBa3EhAAwBCyAAIAJJDQAgACACcCEACyAHIAAgBUYNABogAygCACAAQQJ0aiAENgIAIAEoAgALNgIAIAFBADYCACADIAMoAgxBAWs2AgwgBkEBOgAMIAYgCTYCCCAGIAE2AgQgBigCBCEAIAZBADYCBCAABEAgABDzAQsLIAZBEGokAAsUACAAIAEQwgUiAEGAjwI2AgAgAAukAQECfyMAQRBrIgMkACAAKAIAIQQgAyABKAIANgIIQciKAiADQQhqIgEQCSEAIAMgAioCADgCCAZAQZCLAiABEAkhAQZAIAQgACABEA0ZIAMkAAZAIAEQABkgAyQAEPAFAAsJAAsZIAMkAAZAIAAQABkgAyQAEPAFAAsJAAsGQCABEAAZIAMkABDwBQALBkAgABAAGSADJAAQ8AUACyADQRBqJAALpAEBAn8jAEEQayIDJAAgACgCACEEIAMgASgCADYCCEHIigIgA0EIaiIBEAkhACADIAIoAgA2AggGQEHsigIgARAJIQEGQCAEIAAgARANGSADJAAGQCABEAAZIAMkABDwBQALCQALGSADJAAGQCAAEAAZIAMkABDwBQALCQALBkAgARAAGSADJAAQ8AUACwZAIAAQABkgAyQAEPAFAAsgA0EQaiQAC/YDAgh/A30CQCACQQJJDQAgAkECa0EBdiEIIAAoAgQhBiAAKgIAIQsgACEDA0AgAyIFIARBAWpBA3QiCWohAwJ/IARBAXQiBEEBciIHIAIgBEECaiIKTA0AGiADQQhqIQQgAyoCACIMIAMqAggiDV1FBEAgByAMIA1eDQEaIAcgBSAJaigCBCAEKAIETw0BGgsgBCEDIAoLIQQgBSADKgIAOAIAIAUgAygCBDYCBCAEIAhMDQALIAFBCGsiAiADRgRAIAMgCzgCACADIAY2AgQPCyADIAIqAgA4AgAgAyABQQRrIgEoAgA2AgQgAiALOAIAIAEgBjYCACADIABrQQhqIgFBCUgNAAJAIAAgAUEDdkECayIGQQF2IgFBA3QiBWoiBCoCACIMIAMqAgAiC10EQCADKAIEIQIgACAFaigCBCEFDAELIAsgDF0NASAAIAFBA3RqKAIEIgUgAygCBCICTw0BCyADIAw4AgAgAyAFNgIEAkACQCAGQQJJDQADQAJAIAsgACABQQFrIgZBAXYiAUEDdCIFaiIDKgIAIgxeBEAgACAFaigCBCEFDAELIAsgDF0NAiAAIAVqKAIEIgUgAk8NAgsgBCAFNgIEIAQgDDgCACADIQQgBkEBSw0ACwwBCyAEIQMLIAMgAjYCBCADIAs4AgALC7cJAgZ/AXwjAEEQayIHJAAgAEIANwIEIABCADcDMCAAQZjgADYCACAAQgA3AgwgAEIANwIUIABCADcCHCAAQgA3AiQgAEIANwM4IABBQGtCADcDACAAQcgAaiIKQgA3AwAgAEEANgJQIAdBADoADCAHIAo2AggGQEGAgOAAEKgFIQYZIAckACAHQQhqELMBCQALIAAgBjYCSCAAIAZBgIDgAGoiCTYCUCAGQQBBgIDgABDUARogACAJNgJMIABBADYCdCAAQewAaiIJQgA3AgAgAEIANwJkIABCADcCXCAAQgA3AlQgB0EAOgAMIAcgCTYCCAZAAkAgAgRABkAgAkGr1arVAE8EQBA3DAMLIAJBGGwiCBCoBSEGGSAHJAAgB0EIahCzAQkACyAAIAY2AmwgACAGIAhqNgJ0IAAgBkEAIAhBGGsiBiAGQRhwa0EYaiIGENQBIAZqNgJwCyAAQgA3A3ggAEIANwOYASAAQgA3A5ABIABCADcDiAEgAEIANwOAAQZAIAIEQCAAIAJBAnQiBhCoBSIINgKUASAAIAYgCGoiCzYCnAEgCEEAIAYQ1AEaIAAgCzYCmAELIABCADcDqAEgAEEANgKgASAAQgA3AuwBIABBAToA6AEgAEEANgLkASAAQgE3AtwBIAAgAjYCBCAAQQA2AhQgAEIANwOwASAAQgA3A7gBIABCADcDwAEgAEIANwPIASAAQQA2AtABIABCgICA/BM3AtQBIABCADcC9AEgAEIANwL8ASAAQYQCaiIGQgA3AgAgAEIANwKMAiAAQYCAgPwDNgKUAiAAQcQBaiEIBkAgACABIAEoAgAoAgARAQA2AqABIAAgASABKAIAKAIEEQEANgKkASABIAEoAgAoAggRAQAhASAAQQo2AiggACADNgIcIAAgAzYCGCAAIAE2AqgBIABBADYChAEgACADQQF0NgIgIAAgBCADIAMgBEkbNgIkIAAgA0EDdEEEciIBNgJ8IAAgATYCgAEgAEEBIAVB/////wdwIgMgA0EBTRs2AtgBIAAgACgCoAEgAWoiATYCiAEgAEEBIAVBAWpB/////wdwIgMgA0EBTRs2AtwBIAAgAUEEaiIBNgIMIAAgACgCBCABbBDyASIBNgKMAQJAIAFFBEAGQAZAQQgQ5AUhARgHIAFB2gkQxAUhAQwCGSAHJAAGQCABEOUFGAcJAAsACyAAQQA2AghBNBCoBSEBBkAgASACELQBIQEZIAckACABEPMBCQALIABBfzYCeCAAIAE2AkQgAEF/NgJAIAAgACgCBEECdBDyASIBNgKQAQJAIAFFBEAGQAZAQQgQ5AUhARgIIAFBgRIQxAUhAQwCGSAHJAAGQCABEOUFGAgJAAsACyAAIAAoAhxBAnRBBGo2AhAGQCAAKAIYuBDVASEMGAYgAEQAAAAAAADwPyAMoyIMOQMwIABEAAAAAAAA8D8gDKM5AzggB0EQaiQAIAAPCyABQayQAkEBEOYFDAMLIAFBrJACQQEQ5gUZIAckACAGEIoBIAgQigEgACgClAEiAQRAIAAgATYCmAEgARDzAQsJAAsZIAckACAJELUBCQALCxkgByQAIAoQtQEJAAsAC6oKAQx/IwBBQGoiAyQAIANBADYCLCADQgA3AiQgA0EANgIgIANCADcCGCADQgA3AwggA0IANwMAIANBgICA/AM2AhAgAEEAOgA8IAAoAgRBzAFqIQcDQCABIQYCQAZAAn8DQCAHKAIAIgdFBEAgACgCBEGMAmohCEEAIQFBACEFA0AgCCgCACIIRQRAIAAoAkAiAQRAIAAgATYCRCABEPMBIABBADYCSCAAQgA3AkALIAAgAygCJDYCQCADKAIoIQEgACAJNgJIIAAgATYCRCAAKAJMIgEEQCAAIAE2AlAgARDzASAAQQA2AlQgAEIANwJMCyAAIAMoAhg2AkwgAygCHCEBIAAgCjYCVCAAIAE2AlAgAygCCCICBEADQCACKAIAIQAgAhDzASAAIgINAAsLIAMoAgAhACADQQA2AgAgAARAIAAQ8wELIANBQGskAA8LIAMoAgQiBEUNACADKAIAAn8gCCgCCCIHIARBAWtxIARpQQFLIgxFDQAaIAcgBCAHSw0AGiAHIARwCyILQQJ0aigCACIGRQ0AIAYoAgAiAkUNACABIQYCQCAMRQRAIARBAWshAQNAAkAgByACKAIEIgRHBEAgASAEcSALRg0BIAYhAQwFCyACKAIIIAdGDQMLIAIoAgAiAg0ACyAGIQEMAgsDQAJAIAcgAigCBCIBRwRAIAEgBE8EfyABIARwBSABCyALRg0BIAYhAQwECyACKAIIIAdGDQILIAIoAgAiAg0ACyAGIQEMAQsgAigCDCEEIAUgCkkEQCAFIAQ2AgAgAyAFQQRqIgU2AhwgBiEBDAELIAUgBmsiB0ECdSIFQQFqIgFBgICAgARPBEAQNwwGC0H/////AyAKIAZrIgJBAXYiCiABIAEgCkkbIAJB/P///wdPGyICBH8gAkGAgICABE8EQBCbAQwHCyACQQJ0EKgFBUEACyIBIAVBAnRqIgUgBDYCACADIAEgBiAHENMBIgQgAkECdGoiCjYCICADIAVBBGoiBTYCHCADIAQ2AhggBkUNACAGEPMBDAALAAsgBygCCCEBIAMgB0EMaiICNgI0IANBOGogAyACIANBNGogA0EzahCMASADKAI4IAE2AgwCQCAAKAIEIgQoAogCIgFFDQAgBCgChAICfyACKAIAIgQgAUEBa3EgAWlBAUsiBUUNABogBCABIARLDQAaIAQgAXALIgtBAnRqKAIAIgJFDQAgAigCACICRQ0AIAVFBEAgAUEBayEBA0ACQCAEIAIoAgQiBUcEQCABIAVxIAtGDQEMBAsgAigCCCAERg0ECyACKAIAIgINAAsMAQsDQAJAIAQgAigCBCIFRwRAIAEgBU0EfyAFIAFwBSAFCyALRg0BDAMLIAIoAgggBEYNAwsgAigCACICDQALCyAHKAIIIQQgCCAJSQRAIAggBDYCACADIAhBBGoiCDYCKAwBCwsgCCAGayIIQQJ1IgVBAWoiAUGAgICABE8EQBA3DAMLQQBB/////wMgCSAGayICQQF2IgkgASABIAlJGyACQfz///8HTxsiAkUNABogAkGAgICABE8EQBCbAQwDCyACQQJ0EKgFCyEBGSADJAAgAxCKASADKAIYIgAEQCADIAA2AhwgABDzAQsgAygCJCIABEAgAyAANgIoIAAQ8wELCQALIAEgBUECdGoiBSAENgIAIAMgASAGIAgQ0wEiBCACQQJ0aiIJNgIsIAMgBUEEaiIINgIoIAMgBDYCJCAGRQ0BIAYQ8wEMAQsLAAuDBwEJfyMAQSBrIgMkAAJAIAEgACgCCEkEQAZABkBBCBDkBSEAGAMgAEGFExDEBSEADAIZIAMkACAAEOUFCQALAAsgACgCRCICBEAgAhC/ARDzAQsGQAZAQTQQqAUhAhgCIAIgARC0ASECGSADJAAgAhDzAQkACyAAIAI2AkQCQAJAIAEgACgCmAEgACgClAEiBGtBAnUiAksEQAJAIAEgAmsiAiAAKAKcASIHIAAoApgBIgRrQQJ1TQRAIAAgAgR/IARBACACQQJ0IgIQ1AEgAmoFIAQLNgKYAQwBCwJAIAQgACgClAEiBGsiCEECdSIJIAJqIgZBgICAgARJBEBB/////wMgByAEayIHQQF2IgogBiAGIApJGyAHQfz///8HTxsiBgRAIAZBgICAgARPDQIgBkECdBCoBSEFCyAJQQJ0IAVqQQAgAkECdCICENQBIQcgACAFIAQgCBDTASIFIAZBAnRqNgKcASAAIAIgB2o2ApgBIAAgBTYClAEgBARAIAQQ8wELDAILEDcACxCbAQALIANBADYCFCADQgA3AgwgA0EAOgAcIAMgA0EMajYCGAwBCyABIAJJBEAgACAEIAFBAnRqNgKYAQsgA0EANgIUIANCADcCDCADQQA6ABwgAyADQQxqNgIYIAFFDQELBkAgAUGr1arVAE8EQBA3AAsgAUEYbCICEKgFIQUZIAMkACADQRhqELMBCQALIAMgAiAFajYCFCADIAVBACACQRhrIgIgAkEYcGtBGGoiAhDUASACajYCEAsgAyAAKAJsIgI2AgwgACAFNgJsIAMoAhAhBCADIAAoAnAiBTYCECAAIAQ2AnAgAygCFCEEIAMgACgCdDYCFCAAIAQ2AnQgAgRAIAIhBCACIAVHBEADQCAFQRhrIgUgAkcNAAsgAygCDCEECyADIAI2AhAgBBDzAQsCQCAAKAKMASAAKAIMIAFsEPQBIgJFBEAGQAZAQQgQ5AUhABgEIABBvhcQxAUhAAwCGSADJAAgABDlBQkACwALIAAgAjYCjAECQCAAKAKQASABQQJ0EPQBIgJFBEAGQAZAQQgQ5AUhABgFIABB5RMQxAUhAAwCGSADJAAgABDlBQkACwALIAAgATYCBCAAIAI2ApABIANBIGokAA8LIABBrJACQQEQ5gUACyAAQayQAkEBEOYFAAsgAEGskAJBARDmBQALiQEBAn8jAEEQayICJAACQAJAIAAoAmAgAC0AZyIBIAHAQQBIG0UNAEHs4wItAABFDQAgAEHcAGohAUHt4wItAAAEQCACIAAoAlwgASAALABnQQBIGzYCAEGZ0AAgAhDYAQsgACABEGQMAQtB7eMCLQAARQ0AQZAmEOABCyAAQQE6ADwgAkEQaiQAC+YFAQd/IwBBEGsiByQAIAEoAkgaBkACQAZAAkACQAJAIAEoAsgBIgRFDQAgASgCxAECfyAEQQFrIAJxIARpIgVBAU0NABogAiACIARJDQAaIAIgBHALIgZBAnRqKAIAIgNFDQAgAygCACIDRQ0AAkAgBUEBTQRAIARBAWshBANAAkAgAiADKAIEIgVHBEAgBCAFcSAGRg0BDAULIAMoAgggAkYNAwsgAygCACIDDQALDAILA0ACQCACIAMoAgQiBUcEQCAEIAVNBH8gBSAEcAUgBQsgBkYNAQwECyADKAIIIAJGDQILIAMoAgAiAw0ACwwBCyABKAKEASABKAKMASADKAIMIgIgASgCDGxqai0AAkEBcUUNAQsGQAZAQQgQ5AUhABgGIABB+yUQxAUhAAwCGSAHJAAGQCAAEOUFGAYgB0EAOgAPCQALAAsgASgCqAEoAgAhCSABKAKAASEDIAEoAowBIQUgASgCDCEBQQAhBCAAQQA2AgggAEIANwIAIAMgBSABIAJsamohBUEAIQZBACEDQQAhAgNAAkACQCACIAlJBEAgAyAGRwRAIAMgBSoCADgCACAAIANBBGoiAzYCBAwDCwZAIAYgBGsiA0ECdSIIQQFqIgFBgICAgARPBEAQNwwIC0H/////AyADQQF2IgYgASABIAZJGyADQfz///8HTxsiBkUEQEEAIQEMAwsgBkGAgICABE8EQBCbAQwICyAGQQJ0EKgFIQEMAhkgByQAIAAoAgAiAQRAIAAgATYCBCABEPMBCyAHQQE6AA8JAAsACyAHQRBqJAAPCyABIAhBAnRqIgggBSoCADgCACAAIAEgBCADENMBIgEgBkECdGoiBjYCCCAAIAhBBGoiAzYCBCAAIAE2AgAgBARAIAQQ8wELIAEhBAsgAkEBaiECIAVBBGohBQwACwALIAdBADoADyAAQayQAkEBEOYFGSAHJAAgBy0ADxoJAAsLGSAHJAAJAAsAC+YCAQV/IwBBEGsiBCQAIAAoAkgaBkAGQAJAAkACQCAAKALIASIDRQ0AIAAoAsQBAn8gA0EBayABcSADaSIFQQFNDQAaIAEgASADSQ0AGiABIANwCyIGQQJ0aigCACICRQ0AIAIoAgAiAkUNACAFQQFNBEAgA0EBayEDA0ACQCABIAIoAgQiBUcEQCADIAVxIAZGDQEMBAsgAigCCCABRg0ECyACKAIAIgINAAsMAQsDQAJAIAEgAigCBCIFRwRAIAMgBU0EfyAFIANwBSAFCyAGRg0BDAMLIAIoAgggAUYNAwsgAigCACICDQALCwZABkBBCBDkBSEAGAUgAEH7JRDEBSEADAIZIAQkAAZAIAAQ5QUYBSAEQQA6AA8JAAsACyACKAIMIQEgBEEBOgAPIAAgARDOASAEQRBqJAAPCyAEQQA6AA8gAEGskAJBARDmBRkgBCQAIAQtAA8aCQALGSAEJAAJAAsAC+YCAQV/IwBBEGsiBCQAIAAoAkgaBkAGQAJAAkACQCAAKALIASIDRQ0AIAAoAsQBAn8gA0EBayABcSADaSIFQQFNDQAaIAEgASADSQ0AGiABIANwCyIGQQJ0aigCACICRQ0AIAIoAgAiAkUNACAFQQFNBEAgA0EBayEDA0ACQCABIAIoAgQiBUcEQCADIAVxIAZGDQEMBAsgAigCCCABRg0ECyACKAIAIgINAAsMAQsDQAJAIAEgAigCBCIFRwRAIAMgBU0EfyAFIANwBSAFCyAGRg0BDAMLIAIoAgggAUYNAwsgAigCACICDQALCwZABkBBCBDkBSEAGAUgAEH7JRDEBSEADAIZIAQkAAZAIAAQ5QUYBSAEQQA6AA8JAAsACyACKAIMIQEgBEEBOgAPIAAgARC6ASAEQRBqJAAPCyAEQQA6AA8gAEGskAJBARDmBRkgBCQAIAQtAA8aCQALGSAEJAAJAAsAC8gFAwp/AXwBfSMAQRBrIgIkAAZABkAgASgCACEDQYwfEAwhBBgBIAMgBBARIQMZIAIkAAZAIAQQABkgAiQAEPAFAAsJAAsGQCAEEAAZIAIkABDwBQALBkAgA0HsigIgAkEIahASIQwZIAIkAAZAIAMQABkgAiQAEPAFAAsJAAsGQCACKAIIEBMZIAIkABDwBQALAn8gDEQAAAAAAADwQWMgDEQAAAAAAAAAAGZxBEAgDKsMAQtBAAshCAZAIAMQABkgAiQAEPAFAAsgAEEANgIIIABCADcCAEEAIQQGQAJAIAgEQCAIQYCAgIAETwRAEDcMAgsgACAIQQJ0IgMQqAUiBDYCBCAAIAQ2AgAgACADIARqIgY2AggLIAQhAwNAIAggCU0EQCACQRBqJAAPCyABKAIAIQcgAiAJNgIIQeyKAiACQQhqEAkhBQZAIAcgBRARIQcZIAIkAAZAIAUQABkgAiQAEPAFAAsJAAsGQCAFEAAZIAIkABDwBQALAkAGQAJ/IAdBkIsCIAJBCGoQEiEMBkAgAigCCBATGSACJAAQ8AUACyAMtiENIAMgBkkEQCADIA04AgAgACADQQRqIgM2AgQMAwsgAyAEayILQQJ1IgpBAWoiA0GAgICABE8EQBA3DAULQQBB/////wMgBiAEayIFQQF2IgYgAyADIAZJGyAFQfz///8HTxsiA0UNABogA0GAgICABE8EQBCbAQwFCyADQQJ0EKgFCyEFGSACJAAGQCAHEAAZIAIkABDwBQALCQALIAUgCkECdGoiCiANOAIAIAAgBSAEIAsQ0wEiBSADQQJ0aiIGNgIIIAAgCkEEaiIDNgIEIAAgBTYCACAEBEAgBBDzAQsgBSEECwZAIAcQABkgAiQAEPAFAAsgCUEBaiEJDAALAAsZIAIkACAAKAIAIgEEQCAAIAE2AgQgARDzAQsJAAsAC6UBAQJ/IwBBEGsiASQAIAAoAgQhAiABIAAoAgAiADYCDCABIAIgAGtBAnU2AggGQAZAQcDUACABQQhqEAkhABgBIAAQECECGSABJAAGQCAAEAAZIAEkABDwBQALCQALBkAgABAAGSABJAAQ8AUACwZAIAIQAhkgASQABkAgAhAAGSABJAAQ8AUACwkACwZAIAIQABkgASQAEPAFAAsgAUEQaiQAIAILMQEBf0EEEOQFIgBB0I0CNgIAIABBqI0CNgIAIABBvI0CNgIAIABBqI4CQeUAEOYFAAs/AQJ/IwAhAgZABkBBCBDkBSEBGAEgASAAEMIFIgBBuI8CNgIAGSACJAAgARDlBQkACyAAQdiPAkHjABDmBQALywEBAX8jACECIABBADYCBAZAAkAgAUUEQEHt4wItAAAEQEGbORDgAQsGQAZAQQgQ5AUhARgEIAFBmzkQwgUhAQwCGSACJAAGQCABEOUFGAQJAAsACyAAIAE2AgBBEBCoBSICIAE2AgwgAkHmADYCBCACQeDVADYCACACIAFBAnQ2AgggACACNgIEIAAPCyABQYCPAjYCACABQaSPAkHjABDmBRkgAiQAIAAoAgQhASAAQQA2AgQgAQRAIAEgASgCACgCEBECAAsJAAsAC9gBAgJ9An8gAigCACICRQRAQwAAAAAPCyACQQNxIQUCQCACQQRJBEAMAQsgAkF8cSEGQQAhAgNAIAAqAgwgASoCDJMiAyADlCAAKgIIIAEqAgiTIgMgA5QgACoCBCABKgIEkyIDIAOUIAAqAgAgASoCAJMiAyADlCAEkpKSkiEEIAFBEGohASAAQRBqIQAgAkEEaiICIAZHDQALCyAFBEBBACECA0AgACoCACABKgIAkyIDIAOUIASSIQQgAEEEaiEAIAFBBGohASACQQFqIgIgBUcNAAsLIAQLBwAgACgCCAsHACAAKAIECwcAIABBDGoLBwAgABDzAQvtAQIBfQZ/AkAgAigCACICRQRADAELIAJBA3EhBgJAIAJBBEkEQEEAIQIMAQsgAkF8cSEJQQAhAgNAIAAgAkECdCIEQQxyIgVqKgIAIAEgBWoqAgCUIAAgBEEIciIFaioCACABIAVqKgIAlCAAIARBBHIiBWoqAgAgASAFaioCAJQgACAEaioCACABIARqKgIAlCADkpKSkiEDIAJBBGohAiAIQQRqIgggCUcNAAsLIAZFDQADQCAAIAJBAnQiBGoqAgAgASAEaioCAJQgA5IhAyACQQFqIQIgB0EBaiIHIAZHDQALC0MAAIA/IAOTCwQAIAALggUCAn8BfCMAQUBqIgIkAAJAIAAoAgQiAEEBa0EBTQRAQe3jAi0AAARAQd44EOABCwZABkBBCBDkBSEAGAMgAEHeOBCOASEADAIZIAIkACAAEOUFCQALAAsGQEGQ5AItAABBAXFFBEBBA0HM2QAQASEDQZDkAkEBOgAAQYzkAiADNgIAC0GM5AIoAgAhA0EBEAIgAiABNgIwIAJBATYCKCADIABBtR0gAkEkaiACQShqEBQhBBkgAiQABkBBARAAGSACJAAQ8AUACwZACQEHACEAIAIkAEHkhgNBnAk2AgBB4IYDQQA2AgAgABCVBgJAQeiGAygCAEEBRgRAIAAQ6QUhAEHt4wItAAAEQCACIAAgACgCACgCCBEBADYCAEHWzwAgAhDYAQtBCBDkBSEDIAAgACgCACgCCBEBACEAIAJBAToAPwZAIAJBDGogABAyIQAgAkEBOgA+BkAgAiAAQdHIABDMBSIBKAIINgIgIAIgASkCADcDGCABQgA3AgAgAUEANgIIIAJBAToAPQZAIAMgAkEYahDABSIBQYCPAjYCACACQQA6AD0gAUGkjwJB4wAQ5gUMBBkgAiQAIAItAD0hASACLAAjQQBIBEAgAigCGBDzAQsgAiABQQFxOgA+CQALABkgAiQAIAItAD4hASAALAALQQBIBEAgACgCABDzAQsgAiABQQFxOgA/CQALABkgAiQAIAItAD8EQCADEOUFCwZAEOoFGSACJAAQ8AUACwkACwALCQELAAsACwZAIAIoAiQQExkgAiQAEPAFAAsGQEEBEAAZIAIkABDwBQALIAJBQGskACAERAAAAAAAAAAAYg8LIABBpI8CQeMAEOYFAAvEAwEFfyMAQSBrIgUkACAFIAI2AgwgAEE4aiEIBkACQAJAAkACQCAAKAI8IgZFDQAgCCgCAAJ/IAZBAWsgAnEgBmkiB0EBTQ0AGiACIAIgBkkNABogAiAGcAsiA0ECdGooAgAiBEUNACAEKAIAIgRFDQAgB0EBTQRAIAZBAWshBgNAAkAgAiAEKAIEIgdHBEAgBiAHcSADRg0BDAQLIAQoAgggAkYNBAsgBCgCACIEDQALDAELA0ACQCACIAQoAgQiB0cEQCAGIAdNBH8gByAGcAUgBwsgA0YNAQwDCyAEKAIIIAJGDQMLIAQoAgAiBA0ACwsgACgCDCIEIAAoAghPBEAGQAZAQQgQ5AUhABgGIABBk84AEMQFIQAMBBkgBSQABkAgABDlBRgGCQALAAsgBSAFQQxqIgI2AhQgBUEYaiAIIAIgBUEUaiAFQRNqEIwBIAUoAhggBDYCDCAAIAAoAgxBAWo2AgwMAQsgBCgCDCEECyAAKAIUIAAoAgQgACgCECAEbGpqIAUoAgw2AAAgACgCBCAAKAIQIARsaiABIAAoAhQQ0gEaIAVBIGokAA8LIABBrJACQQEQ5gUZIAUkAAkACwAL5AMCBX8CfSMAQRBrIgckAAZAIAMgASgCDCIFTQRAIABBADYCCCAAQgA3AgACQCAFRQ0AAn0DQAJAIAMgBk0EQCAAKAIAIgYgACgCBEcNAUP//39/DAMLIAIgASgCBCABKAIQIAZsaiABKAIcIAEoAhgRCwAhCiABKAIUIAEoAgQgASgCECAGbGpqKAIAIQUCQCAEBEAgBCAFIAQoAgAoAgARAwBFDQELIAcgBTYCDCAHIAo4AgggACAHQQhqEKgBCyAGQQFqIQYMAQsLIAYqAgALIQogAyEGA0AgBiABKAIMTw0BAkAgAiABKAIEIAEoAhAgBmxqIAEoAhwgASgCGBELACILIApfRQ0AIAEoAhQgASgCBCABKAIQIAZsamooAgAhBQJAIAQEQCAEIAUgBCgCACgCABEDAEUNAQsgByAFNgIMIAcgCzgCCCAAIAdBCGoQqAELIAMgACgCBCIIIAAoAgAiBWtBA3UiCUkEQCAFIAggCRCRASAAIAAoAgRBCGsiCDYCBCAAKAIAIQULIAUgCEYNACAFKgIAIQoLIAZBAWohBgwACwALIAdBEGokAA8LBkBB4Q5BriJB5gBBoRwQFRgBABkgByQAIAAoAgAiAQRAIAAgATYCBCABEPMBCwkACwALngQCB38CfQJAAkACQAJAAkACQCAAKAIEIgIgACgCCCIDSQRAIAIgASkCADcCACAAIAJBCGoiATYCBAwBCyACIAAoAgAiBGtBA3UiCEEBaiIHQYCAgIACTw0CQf////8BIAMgBGsiBUECdiIDIAcgAyAHSxsgBUH4////B08bIgUEfyAFQYCAgIACTw0CIAVBA3QQqAUFQQALIgMgCEEDdGoiBiABKQIANwIAIAZBCGohASACIARHBEADQCAGQQhrIgYgAkEIayICKQIANwIAIAIgBEcNAAsgACgCACECCyAAIAMgBUEDdGo2AgggACABNgIEIAAgBjYCACACRQ0AIAIQ8wEgACgCBCEBCyABIAAoAgAiBGsiAEEJSA0EAkAgBCAAQQN2QQJrIghBAXYiAEEDdCIFaiICKgIAIgkgAUEIayIDKgIAIgpdBEAgAUEEaygCACEHIAQgBWooAgQhBgwBCyAJIApeDQUgBCAAQQN0aigCBCIGIAFBBGsoAgAiB08NBQsgAyAJOAIAIAFBBGsgBjYCACAIQQJJDQIDQAJAIAogBCAAQQFrIgVBAXYiAEEDdCIBaiIDKgIAIgleBEAgASAEaigCBCEBDAELIAkgCl4NBCABIARqKAIEIgEgB08NBAsgAiABNgIEIAIgCTgCACADIQIgBUEBSw0ACwwDCxCbAQALEDcACyACIQMLIAMgBzYCBCADIAo4AgALC64EAQV/IwBBIGsiBSQAIABBADYCCCAAQgA3AgAgASgCACgCBCEGIAVBADYCHAZAIAVBCGogASACIAMgBCAGEQoAIAUoAgwiBCAFKAIIIgFrQQN1IQMGQCABIARHBEBBACECAkAgAyAAKAIIIgcgACgCBCIBa0EDdU0EQCAAIAMEfyABQQAgA0EDdCIBENQBIAFqBSABCzYCBAwBCwJAIAEgACgCACIEa0EDdSIIIANqIgZBgICAgAJJBEBB/////wEgByAEayIHQQJ2IgkgBiAGIAlJGyAHQfj///8HTxsiBgRAIAZBgICAgAJPDQIgBkEDdBCoBSECCyAIQQN0IAJqIgdBACADQQN0IggQ1AEgCGohCCABIARHBEADQCAHQQhrIgcgAUEIayIBKQIANwIAIAEgBEcNAAsgACgCACEBCyAAIAIgBkEDdGo2AgggACAINgIEIAAgBzYCACABBEAgARDzAQsMAgsQNwALEJsBAAsgBSgCDCEECwNAIAQgBSgCCCIBRwRAIAAoAgAgA0EBayIDQQN0aiICIAEqAgA4AgAgAiABKAIENgIEIAEgBCAEIAFrQQN1EJEBIAUgBSgCDEEIayIENgIMDAELCyAEBEAgBSAENgIMIAQQ8wELIAVBIGokAA8ZIAUkACAFKAIIIgEEQCAFIAE2AgwgARDzAQsgBSAAKAIANgIcCQALABkgBSQAIAUoAhwiAQRAIAAgATYCBCABEPMBCwkACwALywEBAn8jAEHAAWsiAiQABkACQAZAIAJBCGogARCrASEBGAIgASAAQQhqQQQQrgIgASAAQRBqQQQQrgIgASAAQQxqQQQQrgIgASAAKAIEIAAoAhAgACgCCGwQrgIgAUEEaiIAEL0CDQAgASABKAIAQQxrKAIAaiIDIAMoAhBBBHIQ3gILGSACJAAgARCsARoJAAsgAUHIlwEoAgAiAzYCACABIANBDGsoAgBqQdSXASgCADYCACAAELwCGiABQegAahCKAiACQcABaiQAC9wBAQR/IwAhAyAAQcyXASgCACICNgIAIABBwJcBNgJoIAAgAkEMaygCAGpB0JcBKAIANgIAIABB6ABqIQQGQAJAIAAgACgCAEEMaygCAGoiAiAAQQRqIgUQ4QIgAkKAgICAcDcCSCAAQcCXATYCaCAAQayXATYCAAZAIAUQugIhAgZAIAIgASgCACABIAEsAAtBAEgbQRQQuQINAiAAIAAoAgBBDGsoAgBqIgEgASgCEEEEchDeAhkgAyQAIAIQvAIaCQALGSADJAAJAAsLGSADJAAgBBCKAgkACyAACzsBAX8gAEHIlwEoAgAiATYCACAAIAFBDGsoAgBqQdSXASgCADYCACAAQQRqELwCGiAAQegAahCKAiAAC1EBAn8gAEHk3AA2AgAgACgCBBDzASAAQUBrKAIAIgIEQANAIAIoAgAhASACEPMBIAEiAg0ACwsgACgCOCEBIABBADYCOCABBEAgARDzAQsgAAtUAQJ/IABB5NwANgIAIAAoAgQQ8wEgAEFAaygCACICBEADQCACKAIAIQEgAhDzASABIgINAAsLIAAoAjghASAAQQA2AjggAQRAIAEQ8wELIAAQ8wELgwUBBn8CQAJAAkACQCABBEAgAUGAgICABE8NASABQQJ0EKgFIQMgACgCACECIAAgAzYCACACBEAgAhDzAQsgACABNgIEQQAhAiABQQRPBEAgAUF8cSEDA0AgAkECdCIGIAAoAgBqQQA2AgAgACgCACAGQQRyakEANgIAIAAoAgAgBkEIcmpBADYCACAAKAIAIAZBDHJqQQA2AgAgAkEEaiECIAVBBGoiBSADRw0ACwsgAUEDcSIDBEADQCAAKAIAIAJBAnRqQQA2AgAgAkEBaiECIARBAWoiBCADRw0ACwsgACgCCCIDRQ0EIABBCGohAiADKAIEIQQgAWkiBUECSQ0CIAEgBE0EQCAEIAFwIQQLIAAoAgAgBEECdGogAjYCACADKAIAIgJFDQQgBUEBTQ0DA0AgASACKAIEIgVNBEAgBSABcCEFCwJAIAQgBUYEQCACIQMMAQsgBUECdCIHIAAoAgBqIgYoAgBFBEAgBiADNgIAIAIhAyAFIQQMAQsgAyACKAIANgIAIAIgACgCACAHaigCACgCADYCACAAKAIAIAdqKAIAIAI2AgALIAMoAgAiAg0ACwwECyAAKAIAIQEgAEEANgIAIAEEQCABEPMBCyAAQQA2AgQMAwsQmwEACyAAKAIAIAQgAUEBa3EiBEECdGogAjYCACADKAIAIgJFDQELIAFBAWshBgNAAkAgBCACKAIEIAZxIgFGBEAgAiEDDAELIAFBAnQiByAAKAIAaiIFKAIABEAgAyACKAIANgIAIAIgACgCACAHaigCACgCADYCACAAKAIAIAdqKAIAIAI2AgAMAQsgBSADNgIAIAIhAyABIQQLIAMoAgAiAg0ACwsL4wEBBH8jACEDIABBsJYBKAIAIgI2AgAgAEGklgE2AmwgACACQQxrKAIAakG0lgEoAgA2AgAgAEEANgIEIABB7ABqIQQGQAJAIAAgACgCAEEMaygCAGoiAiAAQQhqIgUQ4QIgAkKAgICAcDcCSCAAQaSWATYCbCAAQZCWATYCAAZAIAUQugIhAgZAIAIgASgCACABIAEsAAtBAEgbQQwQuQINAiAAIAAoAgBBDGsoAgBqIgEgASgCEEEEchDeAhkgAyQAIAIQvAIaCQALGSADJAAJAAsLGSADJAAgBBCKAgkACyAACzsBAX8gAEGslgEoAgAiATYCACAAIAFBDGsoAgBqQbiWASgCADYCACAAQQhqELwCGiAAQewAahCKAiAAC70GAQV/IwBBIGsiBCQAIABBADoAZyAAIAI2AgAgAEEAOgBcIABCADcCBCAAQgA3AkAgAEIANwIMIABCADcCFCAAQgA3AhwgAEIANwIkIABCADcCLCAAQgA3AjQgAEEAOgA8IABCADcCSCAAQgA3AlAgAEEAOgBYIABBQGshBwZAAkAgAEHcAGoiAiADRg0AIAMsAAtBAE4EQCACIAMpAgA3AgAgAiADKAIINgIIDAELIAMoAgAhBiADKAIEIQMjAEEQayIFJAACQCADQQpNBEAgAiACLQALQYABcSADcjoACyACIAItAAtB/wBxOgALIAIgBiADEJICIAVBADoADyACIANqIAUtAA86AAAMAQsgAkEKIANBCmsgAi0AC0H/AHEiCEEAIAggAyAGEMYFCyAFQRBqJAALAkACQAJAAkAgASgCBCABLQALIgMgA8AiA0EASBtBAmsOBQADAwMBAwsgASgCACABIANBAEgbIgUvAABB7OQARgRAQRAQqAUiASAAKAIAIgI2AgwgAUHmADYCBCABQeDVADYCACABIAJBAnQ2AggMAgsgBS8AAEHp4AFHDQJBEBCoBSIBIAAoAgAiAjYCDCABQeQANgIEIAFBhNgANgIAIAEgAkECdDYCCAwBCyABKAIAIAEgA0EASBtBwSRBBhDXAQ0BQRAQqAUiASAAKAIAIgI2AgwgAUHkADYCBCABQYTYADYCACABIAJBAnQ2AgggAEEBOgBYCyAAIAE2AgggBEEgaiQAIAAPC0Ht4wItAAAEQCAEIAEoAgAgASADQQBIGzYCAEGw0AAgBBDYAQsGQEEIEOQFIQMYASAEQQE6AB8GQCAEQRBqIgUgARDTBSAEQQE6AB4GQCADIAUQwAUiAUGAjwI2AgAgBEEAOgAeIAFBpI8CQeMAEOYFGSAEJAAgBC0AHiEBIAQsABtBAEgEQCAEKAIQEPMBCyAEIAFBAXE6AB8JAAsZIAQkACAELQAfBEAGQCADEOUFGAMLCQALGSAEJAAgACwAZ0EASARAIAIoAgAQ8wELIAAoAkwiAQRAIAAgATYCUCABEPMBCyAHKAIAIgEEQCAAIAE2AkQgARDzAQsJAAsAC1MBBH8CQCAALQAEDQAgACgCACIDKAIAIgFFDQAgASECIAEgAygCBCIERwRAA0AgBEEYayIEIAFHDQALIAAoAgAoAgAhAgsgAyABNgIEIAIQ8wELC6cCAQZ/IwAhBCAAQgA3AgAgACABNgIwIABCADcCKCAAQgA3AiAgAEIANwIYIABCADcCECAAQgA3AghBACEBA0ACQAZAAkAgBUEATARAQQwQqAUiAyAAKAIwIgI2AgggA0H//wM7AQAGQEF/IAJBAXQgAkEASBsQqAUhAgwCGSAEJAAgAxDzAQkACwALIAAPCyADIAI2AgQgAQ0BIAAQtgEZIAQkACAAELcBGgkACyAAKAIQIQELIAAoAgQiAiABQQh2Qfz//wdxaiIGKAIAIgcgAUH/B3FBAnRqQQAgACgCCCACRxsiAiAHRgR/IAZBBGsoAgBBgCBqBSACC0EEayADNgIAIAAgAUEBayIBNgIQIAAgACgCFEEBajYCFCAFQQFqIQUMAAsAC0EBA38gACgCACIBBEAgASECIAEgACgCBCIDRwRAA0AgA0EYayIDIAFHDQALIAAoAgAhAgsgACABNgIEIAIQ8wELC+cKAQx/IwBBEGsiBSQAAkAgACgCCCIBIAAoAgQiB2siA0EIdEEBa0EAIAEgB0cbIAAoAhAiBCAAKAIUamtBgAhPBEAgACAEQYAIajYCECAFIAFBBGsiASgCADYCACAAIAE2AgggACAFEMEBDAELIAMgACgCDCIEIAAoAgAiAWsiAkkEQAJAIAEgB0cEQCAFQYAgEKgFNgIADAELIAVBgCAQqAU2AgACQAJAAkAgACgCCCIBIAAoAgxHBEAgASEEDAELIAAoAgQiAiAAKAIAIghLBEAgACACIAIgCGtBAnVBAWpBfm1BAnQiA2ogAiABIAJrIgEQ0wEgAWoiBDYCCCAAIAAoAgQgA2o2AgQMAQtBASABIAhrQQF1IAEgCEYbIgNBgICAgARPDQEgA0ECdCIEEKgFIgYgBGohByAGIANBfHFqIgMhBAJAIAEgAkYNACABIAJrIgFBfHEhCgJAIAFBBGsiCUECdkEBakEHcSILRQRAIAMhAQwBC0EAIQQgAyEBA0AgASACKAIANgIAIAJBBGohAiABQQRqIQEgBEEBaiIEIAtHDQALCyADIApqIQQgCUEcSQ0AA0AgASACKAIANgIAIAEgAigCBDYCBCABIAIoAgg2AgggASACKAIMNgIMIAEgAigCEDYCECABIAIoAhQ2AhQgASACKAIYNgIYIAEgAigCHDYCHCACQSBqIQIgAUEgaiIBIARHDQALCyAAIAc2AgwgACAENgIIIAAgAzYCBCAAIAY2AgAgCEUNACAIEPMBIAAoAgghBAsgBCAFKAIANgIAIAAgACgCCEEEajYCCAwBCxCbAQALIAUgACgCCEEEayIBKAIANgIAIAAgATYCCAsgACAFEMEBIABBgAQgACgCEEGACGogACgCCCAAKAIEa0EERhs2AhAMAQsGQAJABkBBASACQQF1IAEgBEYbIgNBgICAgARJBEAGQCADQQJ0IgIQqAUhARgFIAUgATYCDAZAQYAgEKgFIQQYAyADBH8gASACagUgBSAENgIIIAUgATYCBEEEEKgFIQMgARDzASAAKAIEIQcgAyIBQQRqCyECIAEgBDYCACABIgMhBgNAIAZBBGohBiAAKAIIIAdGBEAgACACNgIMIAAgBjYCCCAAIAM2AgQgACgCACEEIAAgATYCACAAQYAEIAAoAhBBgAhqIAYgA2tBBEYbNgIQIARFDQQgBBDzAQwECwJAIAIgBkcNACABIANJBEAgAyADIAFrQQJ1QQFqQX5tQQJ0aiADIAIgA2siBBDTASIDIARqIQYMAQtBASACIAFrQQF1IAEgAkYbIgRBgICAgARPBEAgBUEANgIIIAUgATYCBBCbAQALIAVBADYCCCAFIAE2AgQgBEECdCIKEKgFIgggBEF8cWoiBCEGAkAgAiADRg0AIAIgA2siCUF8cSELQQAhBiAEIQIgCUEEayIJQQJ2QQFqQQdxIgwEQANAIAIgAygCADYCACADQQRqIQMgAkEEaiECIAZBAWoiBiAMRw0ACwsgBCALaiEGIAlBHEkNAANAIAIgAygCADYCACACIAMoAgQ2AgQgAiADKAIINgIIIAIgAygCDDYCDCACIAMoAhA2AhAgAiADKAIUNgIUIAIgAygCGDYCGCACIAMoAhw2AhwgA0EgaiEDIAJBIGoiAiAGRw0ACwsgCCAKaiECIAEEQCABEPMBCyAIIQEgBCEDCyAGIAcoAgA2AgAgB0EEaiEHDAALAAsGQBCbARgEABkgBSQAIAUoAgQhACAFKAIIIgEEQCABEPMBCyAFIAA2AgwJAAsACxkgBSQAIAUoAgwiAARAIAAQ8wELCQALCyAFQRBqJAALygEBBH8gAEEANgIUIAAoAggiAiAAKAIEIgFrIgNBCU8EQANAIAEoAgAQ8wEgACAAKAIEQQRqIgE2AgQgACgCCCICIAFrIgNBCEsNAAsLQYAEIQQCQAJAAkAgA0ECdkEBaw4CAQACC0GACCEECyAAIAQ2AhALAkAgASACRg0AA0AgASgCABDzASABQQRqIgEgAkcNAAsgACgCCCIBIAAoAgQiAkYNACAAIAEgAiABa0EDakF8cWo2AggLIAAoAgAiAQRAIAEQ8wELIAALkwMBAn8jAEEgayIEJAAgBCACNgIIAkACQCAALQDoAQ0AIANFDQAGQAZAQQgQ5AUhABgDIABBihYQxAUhAAwCGSAEJAAgABDlBQkACwALIAAoAkgaBkACQCADRQRAIAAgASACELkBDAELBkAgACgCkAIEQCAAKAKMAigCCCEDIARBADoAHyAEIAM2AgQgAEGEAmogBEEEahCNASAEIAAoAogBIAAoAowBIAAoAgwgBCgCBGxqaiIDKAAANgIAIAMgAjYAACAEQQE6AB8gBEEAOgAeBkAgAEHEAWoiAiAEEI0BIARBADoAHiAEKAIEIQMgBCAEQQhqIgU2AhAgBEEUaiACIAUgBEEQaiAEQQ9qEIwBIAQoAhQgAzYCDCAEQQE6AB4gACAEKAIEELoBIARBAToAHiAAIAEgBCgCBBC7AQwDGSAEJAAgBC0AHhogBEEBOgAfCQALAAsgBEEBOgAfIAAgASACELkBGSAEJAAgBC0AHxoJAAsLGSAEJAAJAAsgBEEgaiQADwsgAEGskAJBARDmBQALkREDD38CfQF8IwBBIGsiBCQAIAQgAjYCFCAAQcQBaiEGBkACQAJAAkACQCAAKALIASIFRQ0AIAYoAgACfyAFQQFrIAJxIAVpIgdBAU0NABogAiACIAVJDQAaIAIgBXALIghBAnRqKAIAIgNFDQAgAygCACIDRQ0AAkAgB0EBTQRAIAVBAWshBQNAAkAgAiADKAIEIgdHBEAgBSAHcSAIRg0BDAULIAMoAgggAkYNAwsgAygCACIDDQALDAILA0ACQCACIAMoAgQiB0cEQCAFIAdNBH8gByAFcAUgBwsgCEYNAQwECyADKAIIIAJGDQILIAMoAgAiAw0ACwwBCyADKAIMIQICQAJAIAAtAOgBRQ0AIAAoAoQBIAAoAowBIAAoAgwgAmxqai0AAkEBcUUNAAZABkBBCBDkBSEAGAggAEHCOhDEBSEADAIZIAQkAAZAIAAQ5QUYCCAEQQA6AB8JAAsACyAAKAKEASAAKAKMASAAKAIMIAJsamotAAJBAXEEQCAEQQE6AB8gACACELoBCyAEQQE6AB8gACABIAIQuwEMAgsgBEEAOgAfIABBrJACQQEQ5gUMAwsgACgCCCAAKAIETwRABkAGQEEIEOQFIQAYBiAAQekPEMQFIQAMAxkgBCQABkAgABDlBRgGIARBADoAHwkACwALIAAgACgCCCIKQQFqNgIIIARBADoAHyAEIARBFGoiAjYCGCAEQQRqIAYgAiAEQRhqIAQQjAEgBCgCBCAKNgIMBkAgACgCbBogAEH/////B0EAQf////8HQQAgACgC2AEiAiACQcjbAm4iAkHI2wJsa0GP+QJsIgMgAkHHGmwiAkkbIAMgAmtqIgIgAkHI2wJuIgNByNsCbGtBj/kCbCIFIANBxxpsIgNJGyAFIANraiIDNgLYASADQQFruEQAAID////fQaIgAkEBa7igRAAAAP///89Do0QAAAAAAAAAAKAQ1QEhFBgEIAAoApQBIApBAnRqAn8gACsDMCAUmqIiFJlEAAAAAAAA4EFjBEAgFKoMAQtBgICAgHgLIgk2AgAGQAJAIAAoAkAhByAEIAAoAngiBTYCGCAAKAKEASAAKAKMASAAKAIMIgIgCmxqakEAIAIQ1AEaIAAoAogBIAAoAowBIAAoAgwgCmxqaiAEKAIUNgIAIAAoAoABIAAoAowBIAAoAgwgCmxqaiABIAAoAqABENIBGgZAAkAgCQRAIAAoAhAgCWxBAWoiAhDyASEDIApBAnQiBiAAKAKQAWogAzYCACAAKAKQASAGaigCACIDRQRABkAGQEEIEOQFIQAYCyAAQf4MEMQFIQAMAxkgBCQABkAgABDlBRgLCQALAAsgA0EAIAIQ1AEaCwZAAkACQAJAIAVBf0cEQAJAIAcgCUwNAAZAIAEgACgCgAEgACgCjAEgACgCDCAFbGpqIAAoAqgBIAAoAqQBEQsAIRIYByAHIQIDQCACIAlMDQEgAkEBayECQQEhAwNAIANBAXFFDQEgACgCbBogACgCkAEgBUECdGooAgAgACgCECACbGoiA0EEaiELIAMvAQAhDEEAIQZBACEDA0AgAyAMTwRAIAZBAXEhAwwCCyALIANBAnRqKAIAIgggACgCBEsEQAZABkBBCBDkBSEAGBMgAEGzFxDEBSEADAkZIAQkAAZAIAAQ5QUYEwkACwALIAggBSABIAAoAoABIAAoAowBIAAoAgwgCGxqaiAAKAKoASAAKAKkARELACITIBJdIggbIQUgEyASIAgbIRJBASAGIAgbIQYgA0EBaiEDDAALAAsACwALIAkgByAHIAlKGyELIAAoAoQBIAAoAowBIAAoAgwgBCgCGGxqai0AAkEBcSEPA0AgC0EASA0CIAcgC0gEQAZABkBBCBDkBSEAGBAgAEGnFxDEBSEADAUZIAQkAAZAIAAQ5QUYEAZACQEYCQALAAsGQCAEQQRqIAAgBSABIAsQwgEYBwZAAkAgD0UNACAEIAEgACgCgAEgACgCjAEgACgCDCAEKAIYbGpqIAAoAqgBIAAoAqQBEQsAOAIAIARBBGogBCAEQRhqEMMBIAAoAiQgBCgCCCINIAQoAgQiBmsiAkEDdU8NAAJAIAJBCUkNACACQQN2IhBBAmtBAXYhDCAGKAIEIQ4gBioCACESQQAhAiAGIQMDQCACQQF0IhFBAXIhBSADIgggAkEDdGpBCGohAwJAIBAgEUECaiICTARAIAUhAgwBCyADKgIAIAMqAghdRQRAIAUhAgwBCyADQQhqIQMLIAggAyoCADgCACAIIAMoAgQ2AgQgAiAMTA0ACyANQQhrIgIgA0YEQCADIBI4AgAgAyAONgIEDAELIAMgAioCADgCACADIA1BBGsiBSgCADYCBCACIBI4AgAgBSAONgIAIAMgBmtBCGoiAkEJSA0AIAYgAkEDdkECa0EBdiICQQN0aiIFKgIAIhIgAyoCACITXUUNACADKAIEIQgDQAJAIAMgEjgCACADIAUiAygCBDYCBCACRQ0AIAYgAkEBa0EBdiICQQN0aiIFKgIAIhIgE10NAQsLIAMgCDYCBCADIBM4AgALIAQgDUEIazYCCAsgACAKIARBBGogC0EAEMQBIQUZIAQkACAEKAIEIgAEQCAEIAA2AgggABDzAQsGQAkBGAgACyAEKAIEIgIEQCAEIAI2AgggAhDzAQsgC0EBayELDAALAAsgACAJNgJAIABBADYCeAsgByAJSARAIAAgCTYCQCAAIAo2AngLDAULBkAgAEGskAJBARDmBRgDDAgLIABBrJACQQEQ5gUMBxkgBCQACQALAAsgAEGskAJBARDmBQwFGSAEJAAJAAsACxkgBCQABkAJARgFAAsLIARBIGokAA8LIARBADoAHyAAQayQAkEBEOYFCxkgBCQAIAQtAB8aCQALAAvNAQECfyMAQRBrIgIkACACIAE2AgwCQAJAIAEgACgCCEkEQCAAKAKEASAAKAKMASAAKAIMIAIoAgxsamoiAS0AAiIDQQFxRQ0BIAEgA0H+AXE6AAIgACAAKAIUQQFrNgIUIAAtAOgBRQ0CBkAgAEGEAmogAkEMahCNAQwDGSACJAAJAAsAC0H4DkGxIUG9BkHJHhAVAAsGQAZAQQgQ5AUhABgCIABBzCcQxAUhABkgAiQAIAAQ5QUJAAsgAEGskAJBARDmBQALIAJBEGokAAv5GAIWfwN9IwBBQGoiBCQAIAQgAjYCPCAAKAKAASAAKAKMASAAKAIMIAJsamogASAAKAKgARDSARogACgCQCEVAkAgAiAAKAJ4IhZGBEAgACgCCEEBRg0BIAQoAjwhAgsgACgClAEgAkECdGooAgAhEyAEQTBqIRcgBEEYaiEYA0AgCiATSgRAIAAgASAWIAQoAjwgEyAVEMUBDAILIARCADcDMCAEQgA3AyggBEGAgID8AzYCOCAEQgA3AxggBEIANwMQIARBgICA/AM2AiAGQAJAIAAoAmwaIAQoAjwhAgJ/An8gCkUEQCAAKAKEASAAKAKMASAAKAIMIAJsamoMAQsgACgCkAEgAkECdGooAgAgACgCECAKQQFrbGoLIgMvAQAiAkUEQEEAIQVBACECQQAMAQsGQCACQQJ0IgIQqAUhBRkgBCQACQALIAVBACACENQBIAJqCyEHIAcgBSADQQRqIAIQ0gEiDEYNAAZAIAQgBEEoaiAEQTxqIgIgAhDGASAKQQFrIRQgDCEFA0AgBSAHRgRAIBghDQNAIA0oAgAiDUUNBCAEQQA2AgggBEIANwIAAn8CQAJAIAQoAiwiB0UNACANKAIIIQMCQCAHaUEBSyIFRQRAIAdBAWsgA3EhBgwBCyADIgYgB0kNACADIAdwIQYLIAQoAiggBkECdGooAgAiAkUNACACKAIAIgJFDQAgBUUEQCAHQQFrIQcDQAJAIAMgAigCBCIFRwRAIAUgB3EgBkYNAQwECyACKAIIIANGDQQLIAIoAgAiAg0ACwwBCwNAAkAgAyACKAIEIgVHBEAgBSAHTwR/IAUgB3AFIAULIAZGDQEMAwsgAigCCCADRg0DCyACKAIAIgINAAsLIAQoAjQMAQsgBCgCNEEBawsiAiAAKAIkIgMgAiADSRshDyAXIQgDQAJAAkACQAJ/BkACQCAIKAIAIghFBEAgACAEIABBHEEgIAobaigCABDHASAAKAJsGiANKAIIIQIgCg0BIAAoAoQBIAAoAowBIAAoAgwgAmxqagwDCyAIKAIIIgIgDSgCCCIDRg0GIAAoAoABIgYgACgCjAEiByADIAAoAgwiBWxqaiAHIAIgBWxqIAZqIAAoAqgBIAAoAqQBEQsAIRkgDyAEKAIEIgUgBCgCACIHayILQQN1IgJLBEACQCAEKAIIIgYgBUsEQCAFIBk4AgAgBSAIKAIINgIEIAQgBUEIaiIGNgIEIAchAgwBCyACQQFqIgNBgICAgAJPBEAQNwwIC0H/////ASAGIAdrIgZBAnYiCSADIAMgCUkbIAZB+P///wdPGyIDQYCAgIACTwRAEJsBDAgLIANBA3QiAxCoBSIJIAJBA3RqIgIgGTgCACACIAgoAgg2AgQgAkEIaiEGIAUgB0cEQANAIAJBCGsiAiAFQQhrIgUpAgA3AgAgBSAHRw0ACwsgBCADIAlqNgIIIAQgBjYCBCAEIAI2AgAgB0UNACAHEPMBIAQoAgAhAiAEKAIEIQYLIAYgAmsiA0EJSA0HIAIgA0EDdkECa0EBdiIFQQN0aiIHKgIAIhkgBkEIayIDKgIAIhpdRQ0HIAZBBGsoAgAhBgNAAkAgAyAZOAIAIAMgByIDKAIENgIEIAVFDQAgAiAFQQFrQQF2IgVBA3RqIgcqAgAiGSAaXQ0BCwsgAyAGNgIEIAMgGjgCAAwHCyAZIAcqAgAiGl1FDQYCQCALQQlJDQAgC0EDdiIQQQJrQQF2IREgBygCBCEOQQAhAyAHIQIDQCADQQF0IhJBAXIhBiACIgkgA0EDdGpBCGohAgJAIBAgEkECaiIDTARAIAYhAwwBCyACKgIAIAIqAghdRQRAIAYhAwwBCyACQQhqIQILIAkgAioCADgCACAJIAIoAgQ2AgQgAyARTA0ACyAFQQhrIgMgAkYEQCACIBo4AgAgAiAONgIEDAELIAIgAyoCADgCACACIAVBBGsiBigCADYCBCADIBo4AgAgBiAONgIAIAIgB2tBCGoiA0EJSA0AIAcgA0EDdkECa0EBdiIDQQN0aiIGKgIAIhogAioCACIbXUUNACACKAIEIQkDQAJAIAIgGjgCACACIAYiAigCBDYCBCADRQ0AIAcgA0EBa0EBdiIDQQN0aiIGKgIAIhogG10NAQsLIAIgCTYCBCACIBs4AgALIAQgBUEIayICNgIEIAQoAggiBiACSwRAIAIgGTgCACAFQQRrIAgoAgg2AgAgBCAFNgIEDAULIAIgB2tBA3UiBUEBaiIDQYCAgIACTwRAEDcMBgtB/////wEgBiAHayIGQQJ2IgkgAyADIAlJGyAGQfj///8HTxsiA0GAgICAAk8EQBCbAQwGCyADQQN0IgkQqAUhBgwDCxkgBCQAIAQoAgAiAARAIAQgADYCBCAAEPMBCwkACyAAKAKQASACQQJ0aigCACAAKAIQIBRsagsiAyAEKAIEIgggBCgCACIHa0EDdSICOwEAIAcgCEcEQCADQQRqIQ5BASACIAJBAU0bIQ9BACEJA0AgDiAJQQJ0aiAHKAIENgIAAkAgCCAHayICQQlJDQAgAkEDdiIQQQJrQQF2IREgBygCBCELIAcqAgAhGUEAIQMgByECA0AgA0EBdCISQQFyIQYgAiIFIANBA3RqQQhqIQICQCAQIBJBAmoiA0wEQCAGIQMMAQsgAioCACACKgIIXUUEQCAGIQMMAQsgAkEIaiECCyAFIAIqAgA4AgAgBSACKAIENgIEIAMgEUwNAAsgCEEIayIDIAJGBEAgAiAZOAIAIAIgCzYCBAwBCyACIAMqAgA4AgAgAiAIQQRrIgYoAgA2AgQgAyAZOAIAIAYgCzYCACACIAdrQQhqIgNBCUgNACAHIANBA3ZBAmtBAXYiA0EDdGoiBioCACIZIAIqAgAiGl1FDQAgAigCBCEFA0ACQCACIBk4AgAgAiAGIgIoAgQ2AgQgA0UNACAHIANBAWtBAXYiA0EDdGoiBioCACIZIBpdDQELCyACIAU2AgQgAiAaOAIACyAEIAhBCGsiCDYCBCAJQQFqIgkgD0cNAAsLIAQoAgAiAkUNBCAEIAI2AgQgAhDzAQwECyAGIAVBA3RqIgMgGTgCACADIAgoAgg2AgQgA0EIaiEFIAIgB0cEQANAIANBCGsiAyACQQhrIgIpAgA3AgAgAiAHRw0ACwsgBCAGIAlqNgIIIAQgBTYCBCAEIAM2AgAgBxDzASAEKAIEIgUgBCgCACIHayELCyALQQlIDQEgByALQQN2QQJrQQF2IgNBA3RqIgYqAgAiGSAFQQhrIgIqAgAiGl1FDQEgBUEEaygCACEFA0ACQCACIBk4AgAgAiAGIgIoAgQ2AgQgA0UNACAHIANBAWtBAXYiA0EDdGoiBioCACIZIBpdDQELCyACIAU2AgQgAiAaOAIADAELCwsACyAEIARBKGogBSAFEMYBIABB/////wdBACAAKALcASICIAJByNsCbiICQcjbAmxrQY/5AmwiAyACQccabCICSRsgAyACa2oiAjYC3AECQCACQQFrs0MAAAAwlEMAAAAAkkMAAIA/Xg0AIAQgBEEQaiAFIAUQxgEgACgCbBogBSgCACECAn8CfyAKRQRAIAAoAoQBIAAoAowBIAAoAgwgAmxqagwBCyAAKAKQASACQQJ0aigCACAAKAIQIBRsagsiAy8BACICRQRAQQAhBkEAIQJBAAwBCwZAIAJBAnQiAhCoBSEGGSAEJAAJAAsgBkEAIAIQ1AEgAmoLIgggBiADQQRqIAIQ0gEiAyICRwRAA0AGQCAEIARBKGogAiACEMYBGSAEJAAgAwRAIAMQ8wELCQALIAJBBGoiAiAIRw0ACwsgA0UNACADEPMBCyAFQQRqIQUMAAsAGSAEJAAgDARAIAwQ8wELCQALAAsZIAQkACAEKAIYIgIEQANAIAIoAgAhACACEPMBIAAiAg0ACwsgBCgCECEAIARBADYCECAABEAgABDzAQsgBCgCMCICBEADQCACKAIAIQAgAhDzASAAIgINAAsLIAQoAighACAEQQA2AiggAARAIAAQ8wELCQALIAwEQCAMEPMBCyAEKAIYIgIEQANAIAIoAgAhAyACEPMBIAMiAg0ACwsgBCgCECECIARBADYCECACBEAgAhDzAQsgBCgCMCICBEADQCACKAIAIQMgAhDzASADIgINAAsLIAQoAighAiAEQQA2AiggAgRAIAIQ8wELIApBAWohCgwACwALIARBQGskAAvMLQIUfwR9IwBBIGsiDyQAIABBADYCCCAAQgA3AgAGQAZAAkACQCABKAIIRQ0ABkAgAiABKAKAASABKAKMASABKAJ4IgYgASgCDGxqaiABKAKoASABKAKkARELACEZGAMgASgCQCEMA0AgDEEASgRAIAxBAWshDEEBIQUDQCAFQQFxRQ0CIAEoApABIAZBAnRqKAIAIAEoAhAgDGxqIgUvAQAhCyABIAEoAuQBQQFqNgLkASABIAsgASgC4AFqNgLgASAFQQRqIQlBACEIQQAhBQNAIAUgC08EQCAIQQFxIQUMAgsgCSAFQQJ0aigCACIHIAEoAgRLBEAGQAZAQQgQ5AUhARgKIAFBsxcQxAUhAQwHGSAPJAAGQCABEOUFGAoGQAkBGAkACwALBkAgAiABKAKAASABKAKMASABKAIMIAdsamogASgCqAEgASgCpAERCwAhGhgHIAcgBiAZIBpeIgcbIQYgGiAZIAcbIRlBASAIIAcbIQggBUEBaiEFDAALAAsACwsCQCABKAIUBEAgASgCKCEFIA9BADYCHCAPQQxqIQkgAiEMIAUgAyADIAVJGyERIAQhCyMAQSBrIgckACAHIAY2AhwgASgCRBDIASIULwEAIRIgFCgCBCEVIAlBADYCCCAJQgA3AgAgB0EANgIUIAdCADcCDAJAAkACQAZAAkACQCABKAKMASIEIAEoAgwgBmwiAmoiBSABKAKEAWotAAJBAXENACALBEAgCyAFIAEoAogBaigAACALKAIAKAIAEQMARQ0BIAEoAowBIQQgASgCDCAHKAIcbCECCyAHIAwgASgCgAEgAiAEamogASgCqAEgASgCpAERCwAiGTgCCCAJIAdBCGogB0EcaiICEMMBIAcgByoCCIw4AgQgB0EMaiAHQQRqIAIQwwEMAQsgB0H///97NgIIIAdBDGogB0EIaiAHQRxqEMMBQ///f38hGQsgFSAHKAIcQQF0aiASOwEAA0ACQCAHKAIMIgYgBygCECIKRg0AIAYoAgQhDiAGKgIAIhqMIBleBEAgCSgCBCAJKAIAa0EDdSARRg0BCwJAIAogBmsiAkEJSQ0AIAJBA3YiEEECa0EBdiENQQAhBCAGIQIDQCAEQQF0IhNBAXIhBSACIgggBEEDdGpBCGohAgJAIBAgE0ECaiIETARAIAUhBAwBCyACKgIAIAIqAghdRQRAIAUhBAwBCyACQQhqIQILIAggAioCADgCACAIIAIoAgQ2AgQgBCANTA0ACyAKQQhrIgQgAkYEQCACIBo4AgAgAiAONgIEDAELIAIgBCoCADgCACACIApBBGsiBSgCADYCBCAEIBo4AgAgBSAONgIAIAIgBmtBCGoiBEEJSA0AIAYgBEEDdkECa0EBdiIEQQN0aiIFKgIAIhogAioCACIbXUUNACACKAIEIQgDQAJAIAIgGjgCACACIAUiAigCBDYCBCAERQ0AIAYgBEEBa0EBdiIEQQN0aiIFKgIAIhogG10NAQsLIAIgCDYCBCACIBs4AgALIAcgCkEIazYCECABKAKEASABKAKMASABKAIMIA5samoiEy8BACEQQQEhDiABIAEoAuQBQQFqNgLkASABIBAgASgC4AFqNgLgAQNAIA4gEEsNAiAHIBMgDkECdGooAgAiAjYCCAJAIBUgAkEBdGoiBC8BACASRg0AIAQgEjsBACAHIAwgASgCgAEgASgCjAEgASgCDCACbGpqIAEoAqgBIAEoAqQBEQsAIho4AgQgGSAaXkUgESAJKAIEIAkoAgBrQQN1TXENACAajCEaAkAgBygCECICIAcoAhQiBkkEQCACIBo4AgAgAiAHKAIINgIEIAcgAkEIaiIGNgIQDAELIAIgBygCDCIFa0EDdSIIQQFqIgRBgICAgAJPBEAQNwwIC0H/////ASAGIAVrIgZBAnYiCiAEIAQgCkkbIAZB+P///wdPGyIEQYCAgIACTwRAEJsBDAgLIARBA3QiChCoBSINIAhBA3RqIgQgGjgCACAEIAcoAgg2AgQgBEEIaiEGIAIgBUcEQANAIARBCGsiBCACQQhrIgIpAgA3AgAgAiAFRw0ACwsgByAKIA1qNgIUIAcgBjYCECAHIAQ2AgwgBUUNACAFEPMBIAcoAhAhBgsCQCAGIAcoAgwiCGsiAkEJSA0AIAggAkEDdkECa0EBdiIEQQN0aiIFKgIAIhogBkEIayICKgIAIhtdRQ0AIAZBBGsoAgAhBgNAAkAgAiAaOAIAIAIgBSICKAIENgIEIARFDQAgCCAEQQFrQQF2IgRBA3RqIgUqAgAiGiAbXQ0BCwsgAiAGNgIEIAIgGzgCAAsCQCABKAKMASABKAIMIAcoAghsaiICIAEoAoQBai0AAkEBcQ0AIAsEQCALIAIgASgCiAFqKAAAIAsoAgAoAgARAwBFDQELIAkgB0EEaiAHQQhqEMMBCyARIAkoAgQiCiAJKAIAIgZrIgJBA3VJBEACQCACQQlJDQAgAkEDdiIWQQJrQQF2IRcgBigCBCENIAYqAgAhGkEAIQQgBiECA0AgBEEBdCIYQQFyIQUgAiIIIARBA3RqQQhqIQICQCAWIBhBAmoiBEwEQCAFIQQMAQsgAioCACACKgIIXUUEQCAFIQQMAQsgAkEIaiECCyAIIAIqAgA4AgAgCCACKAIENgIEIAQgF0wNAAsgCkEIayIEIAJGBEAgAiAaOAIAIAIgDTYCBAwBCyACIAQqAgA4AgAgAiAKQQRrIgUoAgA2AgQgBCAaOAIAIAUgDTYCACACIAZrQQhqIgRBCUgNACAGIARBA3ZBAmtBAXYiBEEDdGoiBSoCACIaIAIqAgAiG11FDQAgAigCBCEIA0ACQCACIBo4AgAgAiAFIgIoAgQ2AgQgBEUNACAGIARBAWtBAXYiBEEDdGoiBSoCACIaIBtdDQELCyACIAg2AgQgAiAbOAIACyAJIApBCGsiCjYCBAsgBiAKRg0AIAYqAgAhGQsgDkEBaiEODAALAAsLIAEoAkQhAiACKAIQIgQNAQZAIAIQtgEZIAckAAkACxkgByQAIAcoAgwiAQRAIAcgATYCECABEPMBCyAJKAIAIgEEQCAJIAE2AgQgARDzAQsJAAsgAigCECEECyACKAIEIgUgBEEIdkH8//8HcWoiBigCACILIARB/wdxQQJ0akEAIAIoAgggBUcbIgUgC0YEfyAGQQRrKAIAQYAgagUgBQtBBGsgFDYCACACIARBAWs2AhAgAiACKAIUQQFqNgIUIAcoAgwiAgRAIAcgAjYCECACEPMBCyAHQSBqJAAMAQsACwwBCyABKAIoIQUgD0EANgIcIA9BDGohCSACIQwgBSADIAMgBUkbIREgBCELIwBBIGsiByQAIAcgBiICNgIcIAEoAkQQyAEiFC8BACESIBQoAgQhFSAJQQA2AgggCUIANwIAIAdBADYCFCAHQgA3AgwCQAJAAkAGQAJAAkAgBARAIAsgASgCiAEgASgCjAEgASgCDCACbGpqKAAAIAsoAgAoAgARAwBFDQEgBygCHCECCyAHIAwgASgCgAEgASgCjAEgASgCDCACbGpqIAEoAqgBIAEoAqQBEQsAIho4AgggCSAHQQhqIAdBHGoiAhDDASAHIAcqAgiMOAIEIAdBDGogB0EEaiACEMMBDAELIAdB////ezYCCCAHQQxqIAdBCGogB0EcahDDAUP//39/IRoLIBUgBygCHEEBdGogEjsBAANAAkAgBygCDCIGIAcoAhAiCkYNACAGKAIEIQ4gBioCACIZjCAaXgRAIAtFDQEgCSgCBCAJKAIAa0EDdSARRg0BCwJAIAogBmsiAkEJSQ0AIAJBA3YiEEECa0EBdiENQQAhBSAGIQIDQCAFQQF0IhNBAXIhBCACIgggBUEDdGpBCGohAgJAIBAgE0ECaiIFTARAIAQhBQwBCyACKgIAIAIqAghdRQRAIAQhBQwBCyACQQhqIQILIAggAioCADgCACAIIAIoAgQ2AgQgBSANTA0ACyAKQQhrIgQgAkYEQCACIBk4AgAgAiAONgIEDAELIAIgBCoCADgCACACIApBBGsiBSgCADYCBCAEIBk4AgAgBSAONgIAIAIgBmtBCGoiBEEJSA0AIAYgBEEDdkECa0EBdiIFQQN0aiIEKgIAIhkgAioCACIbXUUNACACKAIEIQgDQAJAIAIgGTgCACACIAQiAigCBDYCBCAFRQ0AIAYgBUEBa0EBdiIFQQN0aiIEKgIAIhkgG10NAQsLIAIgCDYCBCACIBs4AgALIAcgCkEIazYCECABKAKEASABKAKMASABKAIMIA5samoiEy8BACEQQQEhDiABIAEoAuQBQQFqNgLkASABIBAgASgC4AFqNgLgAQNAIA4gEEsNAgJAIBUgEyAOQQJ0aigCACIIQQF0aiICLwEAIBJGDQAgAiASOwEAIAwgASgCgAEgASgCjAEgASgCDCAIbGpqIAEoAqgBIAEoAqQBEQsAIhsgGl1FIBEgCSgCBCAJKAIAa0EDdU1xDQAgG4whGQJAIAcoAhAiAiAHKAIUIgZJBEAgAiAINgIEIAIgGTgCACAHIAJBCGoiBjYCEAwBCyACIAcoAgwiBGtBA3UiCkEBaiIFQYCAgIACTwRAEDcMCAtB/////wEgBiAEayIGQQJ2Ig0gBSAFIA1JGyAGQfj///8HTxsiBUGAgICAAk8EQBCbAQwICyAFQQN0Ig0QqAUiFiAKQQN0aiIFIAg2AgQgBSAZOAIAIAVBCGohBiACIARHBEADQCAFQQhrIgUgAkEIayICKQIANwIAIAIgBEcNAAsLIAcgDSAWajYCFCAHIAY2AhAgByAFNgIMIARFDQAgBBDzASAHKAIQIQYLAkAgBiAHKAIMIgprIgJBCUgNACAKIAJBA3ZBAmtBAXYiBUEDdGoiBCoCACIZIAZBCGsiAioCACIcXUUNACAGQQRrKAIAIQYDQAJAIAIgGTgCACACIAQiAigCBDYCBCAFRQ0AIAogBUEBa0EBdiIFQQN0aiIEKgIAIhkgHF0NAQsLIAIgBjYCBCACIBw4AgALAkACQCALRQ0AIAsgASgCiAEgASgCjAEgASgCDCAIbGpqKAAAIAsoAgAoAgARAwANACAJKAIEIgogCSgCACIGayEIDAELAkAgCSgCBCICIAkoAggiBkkEQCACIAg2AgQgAiAbOAIAIAkgAkEIaiIKNgIEDAELIAIgCSgCACIEa0EDdSIKQQFqIgVBgICAgAJPBEAQNwwJC0H/////ASAGIARrIgZBAnYiDSAFIAUgDUkbIAZB+P///wdPGyIFQYCAgIACTwRAEJsBDAkLIAVBA3QiBhCoBSINIApBA3RqIgUgCDYCBCAFIBs4AgAgBUEIaiEKIAIgBEcEQANAIAVBCGsiBSACQQhrIgIpAgA3AgAgAiAERw0ACwsgCSAGIA1qNgIIIAkgCjYCBCAJIAU2AgAgBEUNACAEEPMBIAkoAgQhCgsgCiAJKAIAIgZrIghBCUgNACAGIAhBA3ZBAmtBAXYiBUEDdGoiBCoCACIZIApBCGsiAioCACIbXUUNACAKQQRrKAIAIQ0DQAJAIAIgGTgCACACIAQiAigCBDYCBCAFRQ0AIAYgBUEBa0EBdiIFQQN0aiIEKgIAIhkgG10NAQsLIAIgDTYCBCACIBs4AgALIBEgCEEDdUkEQAJAIAhBCUkNACAIQQN2IhZBAmtBAXYhFyAGKAIEIQ0gBioCACEZQQAhBSAGIQIDQCAFQQF0IhhBAXIhBCACIgggBUEDdGpBCGohAgJAIBYgGEECaiIFTARAIAQhBQwBCyACKgIAIAIqAghdRQRAIAQhBQwBCyACQQhqIQILIAggAioCADgCACAIIAIoAgQ2AgQgBSAXTA0ACyAKQQhrIgQgAkYEQCACIBk4AgAgAiANNgIEDAELIAIgBCoCADgCACACIApBBGsiBSgCADYCBCAEIBk4AgAgBSANNgIAIAIgBmtBCGoiBEEJSA0AIAYgBEEDdkECa0EBdiIFQQN0aiIEKgIAIhkgAioCACIbXUUNACACKAIEIQgDQAJAIAIgGTgCACACIAQiAigCBDYCBCAFRQ0AIAYgBUEBa0EBdiIFQQN0aiIEKgIAIhkgG10NAQsLIAIgCDYCBCACIBs4AgALIAkgCkEIayIKNgIECyAGIApGDQAgBioCACEaCyAOQQFqIQ4MAAsACwsgASgCRCECIAIoAhAiBQ0BBkAgAhC2ARkgByQACQALGSAHJAAgBygCDCIBBEAgByABNgIQIAEQ8wELIAkoAgAiAQRAIAkgATYCBCABEPMBCwkACyACKAIQIQULIAIoAgQiBCAFQQh2Qfz//wdxaiIGKAIAIgsgBUH/B3FBAnRqQQAgAigCCCAERxsiBCALRgR/IAZBBGsoAgBBgCBqBSAEC0EEayAUNgIAIAIgBUEBazYCECACIAIoAhRBAWo2AhQgBygCDCICBEAgByACNgIQIAIQ8wELIAdBIGokAAwBCwALCyADIA8oAhAiDCAPKAIMIgRrIgVBA3VJBEADQAJAIAVBCUkNACAFQQN2IgdBAmtBAXYhCSAEKAIEIQggBCoCACEZQQAhAiAEIQUDQCACQQF0IgpBAXIhBiAFIgsgAkEDdGpBCGohBQJAIAcgCkECaiICTARAIAYhAgwBCyAFKgIAIAUqAghdRQRAIAYhAgwBCyAFQQhqIQULIAsgBSoCADgCACALIAUoAgQ2AgQgAiAJTA0ACyAMQQhrIgIgBUYEQCAFIBk4AgAgBSAINgIEDAELIAUgAioCADgCACAFIAxBBGsiBigCADYCBCACIBk4AgAgBiAINgIAIAUgBGtBCGoiAkEJSA0AIAQgAkEDdkECa0EBdiICQQN0aiIGKgIAIhkgBSoCACIaXUUNACAFKAIEIQsDQAJAIAUgGTgCACAFIAYiBSgCBDYCBCACRQ0AIAQgAkEBa0EBdiICQQN0aiIGKgIAIhkgGl0NAQsLIAUgCzYCBCAFIBo4AgALIAxBCGsiDCAEayIFQQN1IANLDQALCwNAIAQgDEcEQCAEKgIAIRkgDyABKAKIASABKAKMASABKAIMIAQoAgRsamooAAA2AhAgDyAZOAIMIA8gBDYCHCAAIA9BDGoQqAECQCAMIARrIgJBCUkNACACQQN2IghBAmtBAXYhByAEKAIEIQsgBCoCACEZQQAhAiAEIQUDQCACQQF0IglBAXIhAyAFIgYgAkEDdGpBCGohBQJAIAggCUECaiICTARAIAMhAgwBCyAFKgIAIAUqAghdRQRAIAMhAgwBCyAFQQhqIQULIAYgBSoCADgCACAGIAUoAgQ2AgQgAiAHTA0ACyAMQQhrIgIgBUYEQCAFIBk4AgAgBSALNgIEIAxBCGshDAwDCyAFIAIqAgA4AgAgBSAMQQRrIgMoAgA2AgQgAiAZOAIAIAMgCzYCACAFIARrQQhqIgJBCUgNACAEIAJBA3ZBAmtBAXYiAkEDdGoiAyoCACIZIAUqAgAiGl1FDQAgBSgCBCEGA0ACQCAFIBk4AgAgBSADIgUoAgQ2AgQgAkUNACAEIAJBAWtBAXYiAkEDdGoiAyoCACIZIBpdDQELCyAFIAY2AgQgBSAaOAIACyAMQQhrIQwMAQsLIARFDQAgBBDzAQsgD0EgaiQADwsZIA8kACAPKAIcIgEEQCABEPMBCwkACyABQayQAkEBEOYFGSAPJAAgACgCACIBBEAgACABNgIEIAEQ8wELCQALAAu0AwEEfyMAQcABayIDJAAGQAZAIANBCGogARCrASECGAEgAiAAQYQBakEEEK4CIAIgAEEEakEEEK4CIAIgAEEIakEEEK4CIAIgAEEMakEEEK4CIAIgAEGIAWpBBBCuAiACIABBgAFqQQQQrgIgAiAAQUBrQQQQrgIgAiAAQfgAakEEEK4CIAIgAEEcakEEEK4CIAIgAEEgakEEEK4CIAIgAEEYakEEEK4CIAIgAEEwakEIEK4CIAIgAEEkakEEEK4CIAIgACgCjAEgACgCDCAAKAIIbBCuAkEAIQEDQCAAKAIIIAFNBEAgAkEEaiIAEL0CRQRAIAIgAigCAEEMaygCAGoiASABKAIQQQRyEN4CCyACQciXASgCACIBNgIAIAIgAUEMaygCAGpB1JcBKAIANgIAIAAQvAIaIAJB6ABqEIoCIANBwAFqJAAPCyADIAFBAnQiBSAAKAKUAWooAgAiBCAAKAIQbEEAIARBAEobNgIEIAIgA0EEakEEEK4CIAMoAgQiBARAIAIgACgCkAEgBWooAgAgBBCuAgsgAUEBaiEBDAALABkgAyQAIAIQrAEaCQALAAv5AgEDfyAAQZjgADYCACAAKAKMARDzASAAKAIIBEADQCABQQJ0IgIgACgClAFqKAIAQQBKBEAgACgCkAEgAmooAgAQ8wELIAFBAWoiASAAKAIISQ0ACwsgACgCkAEQ8wEgACgCRCIBBEAgARC/ARDzAQsgACgCjAIiAQRAA0AgASgCACECIAEQ8wEgAiIBDQALCyAAKAKEAiEBIABBADYChAIgAQRAIAEQ8wELIAAoAswBIgEEQANAIAEoAgAhAiABEPMBIAIiAQ0ACwsgACgCxAEhASAAQQA2AsQBIAEEQCABEPMBCyAAKAKUASIBBEAgACABNgKYASABEPMBCyAAKAJsIgMEQCADIQIgAyAAKAJwIgFHBEADQCABQRhrIgEgA0cNAAsgACgCbCECCyAAIAM2AnAgAhDzAQsgACgCSCIDBEAgAyECIAMgACgCTCIBRwRAA0AgAUEYayIBIANHDQALIAAoAkghAgsgACADNgJMIAIQ8wELIAALpAEBBH8gACgCFCIBBEADQCAAKAIEIgQgACgCECIDQQh2Qfz//wdxaigCACADQf8HcUECdGooAgAhAiAAIAFBAWs2AhQgACADQQFqIgE2AhAgAUGAEE8EQCAEKAIAEPMBIAAgACgCBEEEajYCBCAAIAAoAhBBgAhrNgIQCyACBEAgAigCBCIBBEAgARDzAQsgAhDzAQsgACgCFCIBDQALCyAAELcBCwoAIAAQvgEQ8wEL2QMBCn8CQAJAIAAoAgQiBSAAKAIARwRAIAUhAwwBCyAAKAIIIgYgACgCDCIDSQRAIAYgAyAGa0ECdUEBakECbUECdCIEaiEDIAUgBkcEQCADIAYgBWsiAmsiAyAFIAIQ0wEaIAAoAgghBQsgACADNgIEIAAgBCAFajYCCAwBC0EBIAMgBWtBAXUgAyAFRhsiAkGAgICABE8NASACQQJ0IgMQqAUiCCADaiEJIAggAkEDakF8cWoiAyEHAkAgBSAGRg0AIAYgBWsiBkF8cSEKIAMhBCAFIQIgBkEEayILQQJ2QQFqQQdxIgYEQEEAIQcDQCAEIAIoAgA2AgAgAkEEaiECIARBBGohBCAHQQFqIgcgBkcNAAsLIAMgCmohByALQRxJDQADQCAEIAIoAgA2AgAgBCACKAIENgIEIAQgAigCCDYCCCAEIAIoAgw2AgwgBCACKAIQNgIQIAQgAigCFDYCFCAEIAIoAhg2AhggBCACKAIcNgIcIAJBIGohAiAEQSBqIgQgB0cNAAsLIAAgCTYCDCAAIAc2AgggACADNgIEIAAgCDYCACAFRQ0AIAUQ8wEgACgCBCEDCyADQQRrIAEoAgA2AgAgACAAKAIEQQRrNgIEDwsQmwEAC9EPAhF/A30jAEEgayIGJAAgBiACNgIcIAEoAkQQyAEiEC8BACENIBAoAgQhESAAQQA2AgggAEIANwIAIAZBADYCFCAGQgA3AgwCQAJABkACQCABKAKMASABKAIMIAJsaiICIAEoAoQBai0AAkEBcUUEQCAGIAMgAiABKAKAAWogASgCqAEgASgCpAERCwA4AgggACAGQQhqIAZBHGoiAhDDASAGIAYqAggiGIw4AgQgBkEMaiAGQQRqIAIQwwEMAQsgBkH///97NgIIIAZBDGogBkEIaiAGQRxqEMMBQ///f38hGAsgESAGKAIcQQF0aiANOwEAIARBAWshEgNAAkAgBigCDCIHIAYoAhAiC0YNACAHKAIEIQkgByoCACIWjCAYXgRAIAEoAiQgACgCBCAAKAIAa0EDdUYNAQsCQCALIAdrIgJBCUkNACACQQN2IgxBAmtBAXYhDkEAIQUgByECA0AgBUEBdCIPQQFyIQggAiIKIAVBA3RqQQhqIQICQCAMIA9BAmoiBUwEQCAIIQUMAQsgAioCACACKgIIXUUEQCAIIQUMAQsgAkEIaiECCyAKIAIqAgA4AgAgCiACKAIENgIEIAUgDkwNAAsgC0EIayIFIAJGBEAgAiAWOAIAIAIgCTYCBAwBCyACIAUqAgA4AgAgAiALQQRrIggoAgA2AgQgBSAWOAIAIAggCTYCACACIAdrQQhqIgVBCUgNACAHIAVBA3ZBAmtBAXYiBUEDdGoiCCoCACIWIAIqAgAiF11FDQAgAigCBCEKA0ACQCACIBY4AgAgAiAIIgIoAgQ2AgQgBUUNACAHIAVBAWtBAXYiBUEDdGoiCCoCACIWIBddDQELCyACIAo2AgQgAiAXOAIACyAGIAtBCGs2AhAgASgCbBoCfyAERQRAIAEoAoQBIAEoAowBIAEoAgwgCWxqagwBCyABKAKQASAJQQJ0aigCACABKAIQIBJsagsiAkEEaiEOIAIvAQAhD0EAIQsDQAJAAkAgCyAPSQRAIAYgDiALQQJ0aigCACICNgIIIBEgAkEBdGoiBS8BACANRg0CIAUgDTsBAAZAIAYgAyABKAKAASABKAKMASABKAIMIAJsamogASgCqAEgASgCpAERCwAiFjgCBCAWIBhdRSABKAIkIAAoAgQgACgCAGtBA3VNcQ0DIBaMIRYCQCAGKAIQIgIgBigCFCIHSQRAIAIgFjgCACACIAYoAgg2AgQgBiACQQhqIgc2AhAMAQsgAiAGKAIMIghrQQN1IgpBAWoiBUGAgICAAk8EQBA3DAsLQf////8BIAcgCGsiB0ECdiIJIAUgBSAJSRsgB0H4////B08bIgVBgICAgAJPBEAQmwEMCwsgBUEDdCIJEKgFIgwgCkEDdGoiBSAWOAIAIAUgBigCCDYCBCAFQQhqIQcgAiAIRwRAA0AgBUEIayIFIAJBCGsiAikCADcCACACIAhHDQALCyAGIAkgDGo2AhQgBiAHNgIQIAYgBTYCDCAIRQ0AIAgQ8wEgBigCECEHCwJAIAcgBigCDCIKayICQQlIDQAgCiACQQN2QQJrQQF2IgVBA3RqIggqAgAiFiAHQQhrIgIqAgAiF11FDQAgB0EEaygCACEHA0ACQCACIBY4AgAgAiAIIgIoAgQ2AgQgBUUNACAKIAVBAWtBAXYiBUEDdGoiCCoCACIWIBddDQELCyACIAc2AgQgAiAXOAIACyABKAKEASABKAKMASABKAIMIAYoAghsamotAAJBAXENAiAAIAZBBGogBkEIahDDAQwCGSAGJAAJAAsACwwECyABKAIkIAAoAgQiCSAAKAIAIgdrIgJBA3VJBEACQCACQQlJDQAgAkEDdiITQQJrQQF2IRQgBygCBCEMIAcqAgAhFkEAIQUgByECA0AgBUEBdCIVQQFyIQggAiIKIAVBA3RqQQhqIQICQCATIBVBAmoiBUwEQCAIIQUMAQsgAioCACACKgIIXUUEQCAIIQUMAQsgAkEIaiECCyAKIAIqAgA4AgAgCiACKAIENgIEIAUgFEwNAAsgCUEIayIFIAJGBEAgAiAWOAIAIAIgDDYCBAwBCyACIAUqAgA4AgAgAiAJQQRrIggoAgA2AgQgBSAWOAIAIAggDDYCACACIAdrQQhqIgVBCUgNACAHIAVBA3ZBAmtBAXYiBUEDdGoiCCoCACIWIAIqAgAiF11FDQAgAigCBCEKA0ACQCACIBY4AgAgAiAIIgIoAgQ2AgQgBUUNACAHIAVBAWtBAXYiBUEDdGoiCCoCACIWIBddDQELCyACIAo2AgQgAiAXOAIACyAAIAlBCGsiCTYCBAsgByAJRg0AIAcqAgAhGAsgC0EBaiELDAALAAsLIAEoAkQhASABKAIQIgUNAQZAIAEQtgEZIAYkAAkACxkgBiQAIAYoAgwiAQRAIAYgATYCECABEPMBCyAAKAIAIgEEQCAAIAE2AgQgARDzAQsJAAsgASgCECEFCyABKAIEIgAgBUEIdkH8//8HcWoiAigCACIDIAVB/wdxQQJ0akEAIAEoAgggAEcbIgAgA0YEfyACQQRrKAIAQYAgagUgAAtBBGsgEDYCACABIAVBAWs2AhAgASABKAIUQQFqNgIUIAYoAgwiAARAIAYgADYCECAAEPMBCyAGQSBqJAAPCwALpwMCBn8CfQJAAkACQCAAKAIEIgQgACgCCCIGSQRAIAQgASoCADgCACAEIAIoAgA2AgQgACAEQQhqIgE2AgQMAQsgBCAAKAIAIgVrQQN1IghBAWoiA0GAgICAAk8NAUH/////ASAGIAVrIgZBAnYiByADIAMgB0kbIAZB+P///wdPGyIDQYCAgIACTw0CIANBA3QiBhCoBSIHIAhBA3RqIgMgASoCADgCACADIAIoAgA2AgQgA0EIaiEBIAQgBUcEQANAIANBCGsiAyAEQQhrIgQpAgA3AgAgBCAFRw0ACwsgACAGIAdqNgIIIAAgATYCBCAAIAM2AgAgBUUNACAFEPMBIAAoAgQhAQsCQCABIAAoAgAiAmsiAEEJSA0AIAIgAEEDdkECa0EBdiIDQQN0aiIAKgIAIgkgAUEIayIEKgIAIgpdRQ0AIAFBBGsoAgAhBQNAAkAgBCAJOAIAIAQgACIBKAIENgIEIANFDQAgASEEIAIgA0EBa0EBdiIDQQN0aiIAKgIAIgkgCl0NAQsLIAEgBTYCBCABIAo4AgALDwsQNwALEJsBAAuTGAIQfwJ9IwBBMGsiBiQAIAYgATYCKCAAQRxBICADG2ooAgAhESAAIAIgACgCGBDHAQJAIAAoAhgiASACKAIEIgogAigCACIJa0EDdUkEQAZABkBBCBDkBSEAGAMgAEHiKRDEBSEADAIZIAYkACAAEOUFCQALAAsgBkEANgIkIAZCADcCHAZAAkAgAQRAIAFBgICAgARPBEAQNwwCCyAGIAFBAnQiARCoBSILNgIgIAYgCzYCHCAGIAEgC2oiDTYCJAsgCyEHA0AgCSAKRwRAAkAgByANRwRAIAcgCSgCBDYCACAGIAdBBGoiBzYCIAwBCyANIAtrIgFBAnUiCEEBaiIFQYCAgIAETwRAEDcMBAtB/////wMgAUEBdiIHIAUgBSAHSRsgAUH8////B08bIgUEfyAFQYCAgIAETwRAEJsBDAULIAVBAnQQqAUFQQALIgcgCEECdGoiCCAJKAIENgIAIAYgByALIAEQ0wEiASAFQQJ0aiINNgIkIAYgCEEEaiIHNgIgIAYgATYCHCALBEAgCxDzASACKAIEIQogAigCACEJCyABIQsLAkAgCiAJayIBQQlJDQAgAUEDdiISQQJrQQF2IRMgCSgCBCEOIAkqAgAhFUEAIQUgCSEBA0AgBUEBdCIPQQFyIQggASIMIAVBA3RqQQhqIQECQCASIA9BAmoiBUwEQCAIIQUMAQsgASoCACABKgIIXUUEQCAIIQUMAQsgAUEIaiEBCyAMIAEqAgA4AgAgDCABKAIENgIEIAUgE0wNAAsgCkEIayIFIAFGBEAgASAVOAIAIAEgDjYCBAwBCyABIAUqAgA4AgAgASAKQQRrIggoAgA2AgQgBSAVOAIAIAggDjYCACABIAlrQQhqIgVBCUgNACAJIAVBA3ZBAmtBAXYiBUEDdGoiCCoCACIVIAEqAgAiFl1FDQAgASgCBCEMA0ACQCABIBU4AgAgASAIIgEoAgQ2AgQgBUUNACAJIAVBAWtBAXYiBUEDdGoiCCoCACIVIBZdDQELCyABIAw2AgQgASAWOAIACyACIApBCGsiCjYCBAwBCwsgACgCbCECIAYoAigaIAYoAiAiAUEEaygCACESQQAhCgZAIAQEQCACRQRAIAZBADoALyMAQRBrIgAkAEEQEOQFIQFBuYIDLQAARQRAQbmCA0EBOgAACyAAQcDbAjYCDCAAQT82AgggACAAKQIINwMABkAgASAAQe4KEOMFGhkgACQAIAEQ5QUJAAsgAUGshQJB0wMQ5gUACyAGQQA6AC9BASEKCyAGKAIoIQICQAJAAn8gA0UEQCAAKAKEASAAKAKMASAAKAIMIAJsamoMAQsgACgCkAEgAkECdGooAgAgACgCECADQQFrbGoLIgUoAgBFDQAgBA0ABkAGQEEIEOQFIQAYByAAQbYNEMQFIQAMAhkgBiQABkAgABDlBRgHIAYgCjoALwkACwALIAUgASAGKAIcIgJrQQJ1Igg7AQACQCABIAJGDQAgBUEEaiEFQQEgCCAIQQFNGyEIIAAoApQBIQtBACEBAkACQCAEBEADQCALIAIgAUECdCIHaigCACIJQQJ0aigCACADSA0CIAUgB2ogCTYCACABQQFqIgEgCEcNAAwECwALA0AgBSABQQJ0IgdqIgkoAgAEQAZABkBBCBDkBSEAGAsgAEGDGxDEBSEADAQZIAYkAAZAIAAQ5QUYCyAGIAo6AC8JAAsACyALIAIgB2ooAgAiB0ECdGooAgAgA0gNASAJIAc2AgAgCCABQQFqIgFHDQALDAILBkAGQEEIEOQFIQAYCCAAQfwdEMQFIQAZIAYkAAZAIAAQ5QUYCCAGIAo6AC8JAAsgBiAKOgAvIABBrJACQQEQ5gUMBAsgBiAKOgAvIABBrJACQQEQ5gUMAwsgA0EBayETQQAhAgZAAkACQANAIAYoAiAgBigCHCIBa0ECdSACTQRAIAEEQCAGIAE2AiAgARDzAQsgBkEwaiQAIBIPCyAAKAJsGiABIAJBAnQiDmoiASgCABogASgCACEFAn8gA0UEQCAAKAKEASAAKAKMASAAKAIMIAVsamoMAQsgACgCkAEgBUECdGooAgAgACgCECATbGoLIg0vAQAiCyARSwRABkAGQEEIEOQFIQAYCyAAQY4YEMQFIQAMBBkgBiQABkAgABDlBRgLCQALAAsgBigCKCIIIAVGBEAGQAZAQQgQ5AUhABgLIABBzSMQxAUhAAwDGSAGJAAGQCAAEOUFGAsJAAsACwJAIAMgACgClAEgBUECdGooAgBKBEAGQAZAQQgQ5AUhABgMIABB/B0QxAUhAAwCGSAGJAAGQCAAEOUFGAwJAAsACyANQQRqIQwCQAJAIARFDQBBACEBIAtFDQADQCAMIAFBAnRqKAIAIAhGDQIgAUEBaiIBIAtHDQALCyALIBFJBEAgDCALQQJ0aiAINgIAIA0gC0EBajsBAAwBCyAGIAAoAoABIgEgACgCjAEiByAIIAAoAgwiCWxqaiAHIAUgCWxqIAFqIAAoAqgBIAAoAqQBEQsAOAIYQQAhCiAGQQA2AhAgBkIANwIIBkACQCAGQQhqIAZBGGogBkEoahDDAQNAIAogC08EQCAAIAZBCGogERDHAUEAIQsgBigCDCIKIAYoAggiB0cEQANAIAwgC0ECdGogBygCBDYCAAJAIAogB2siAUEJSQ0AIAFBA3YiD0ECa0EBdiEQIAcoAgQhDiAHKgIAIRVBACEFIAchAQNAIAVBAXQiFEEBciEIIAEiCSAFQQN0akEIaiEBAkAgDyAUQQJqIgVMBEAgCCEFDAELIAEqAgAgASoCCF1FBEAgCCEFDAELIAFBCGohAQsgCSABKgIAOAIAIAkgASgCBDYCBCAFIBBMDQALIApBCGsiBSABRgRAIAEgFTgCACABIA42AgQMAQsgASAFKgIAOAIAIAEgCkEEayIIKAIANgIEIAUgFTgCACAIIA42AgAgASAHa0EIaiIFQQlIDQAgByAFQQN2QQJrQQF2IgVBA3RqIggqAgAiFSABKgIAIhZdRQ0AIAEoAgQhCQNAAkAgASAVOAIAIAEgCCIBKAIENgIEIAVFDQAgByAFQQFrQQF2IgVBA3RqIggqAgAiFSAWXQ0BCwsgASAJNgIEIAEgFjgCAAsgBiAKQQhrIgo2AgwgC0EBaiELIAcgCkcNAAsLIA0gCzsBACAHRQ0CIAYgBzYCDCAHEPMBDAILIAAoAoABIgEgACgCjAEiBSAAKAIMIgggDCAKQQJ0aiIHKAIAbGpqIAUgBigCHCAOaigCACAIbGogAWogACgCqAEgACgCpAERCwAhFQJAIAYoAgwiASAGKAIQIglJBEAgASAVOAIAIAEgBygCADYCBCAGIAFBCGoiBzYCDAwBCyABIAYoAggiCGtBA3UiD0EBaiIFQYCAgIACTwRAEDcMDQtB/////wEgCSAIayIJQQJ2IhAgBSAFIBBJGyAJQfj///8HTxsiBUGAgICAAk8EQBCbAQwNCyAFQQN0IgkQqAUiECAPQQN0aiIFIBU4AgAgBSAHKAIANgIEIAVBCGohByABIAhHBEADQCAFQQhrIgUgAUEIayIBKQIANwIAIAEgCEcNAAsLIAYgCSAQajYCECAGIAc2AgwgBiAFNgIIIAhFDQAgCBDzASAGKAIMIQcLAkAgByAGKAIIIglrIgFBCUgNACAJIAFBA3ZBAmtBAXYiBUEDdGoiCCoCACIVIAdBCGsiASoCACIWXUUNACAHQQRrKAIAIQcDQAJAIAEgFTgCACABIAgiASgCBDYCBCAFRQ0AIAkgBUEBa0EBdiIFQQN0aiIIKgIAIhUgFl0NAQsLIAEgBzYCBCABIBY4AgALIApBAWohCgwACwALGSAGJAAgBigCCCIABEAgBiAANgIMIAAQ8wELCQALCyACQQFqIQIMAQsLIABBrJACQQEQ5gUMBQsgAEGskAJBARDmBQwECyAAQayQAkEBEOYFDAMZIAYkAAZACQEYBAALAAsgBiAKOgAvIABBrJACQQEQ5gUZIAYkACAGLQAvGgkACwsZIAYkACAGKAIcIgAEQCAGIAA2AiAgABDzAQsJAAsACyAAQayQAkEBEOYFAAvADgIMfwJ9IwBBMGsiCCQAIAggAjYCLCAEIAVIBEAgASAAKAKAASAAKAKMASAAKAIMIAJsamogACgCqAEgACgCpAERCwAhEiAFIQsDQCALIgdBAWshCwNAIAAoAmwaAn8gB0UEQCAAKAKEASAAKAKMASAAKAIMIAJsamoMAQsgACgCkAEgAkECdGooAgAgACgCECALbGoLIgYvAQAiDARAIAZBBGohDUEAIQZBACEKA0AGQCABIAAoAoABIAAoAowBIA0gBkECdGooAgAiCSAAKAIMbGpqIAAoAqgBIAAoAqQBEQsAIRMZIAgkAAkACyAJIAIgEiATXiIJGyECIBMgEiAJGyESQQEgCiAJGyEKIAZBAWoiBiAMRw0ACyAKQQFxDQELCyAEIAtIDQALCyAEIAVMBEADQAJAAkACQCAEQQBOBEAgCEEcaiAAIAIgASAEEMIBQQAhCyAIQQA2AhQgCEIANwIMIAgoAhwhBSAIKAIgIQZBACEJBkADQCAFIAZHBEAgAyAFKAIERwRAAkAgCCgCFCAJRwRAIAkgBSkCADcCACAIIAlBCGoiBTYCEAwBCyAJIAtrIgZBA3UiCkEBaiIHQYCAgIACTwRAEDcMCQtB/////wEgBkECdiIMIAcgByAMSRsgBkH4////B08bIgcEfyAHQYCAgIACTwRAEJsBDAoLIAdBA3QQqAUFQQALIgwgCkEDdGoiBiAFKQIANwIAIAZBCGohBSAJIAtHBEADQCAGQQhrIgYgCUEIayIJKQIANwIAIAkgC0cNAAsgCCgCDCELCyAIIAwgB0EDdGo2AhQgCCAFNgIQIAggBjYCDCALRQ0AIAsQ8wEgCCgCECEFCwJAIAUgCCgCDCILayIGQQlIDQAgCyAGQQN2QQJrQQF2IgdBA3RqIgoqAgAiEiAFQQhrIgYqAgAiE11FDQAgBUEEaygCACEJA0ACQCAGIBI4AgAgBiAKIgYoAgQ2AgQgB0UNACALIAdBAWtBAXYiB0EDdGoiCioCACISIBNdDQELCyAGIAk2AgQgBiATOAIACyAFIQkLAkAgCCgCICINIAgoAhwiBWsiBkEJSQ0AIAZBA3YiD0ECa0EBdiEQIAUoAgQhDiAFKgIAIRJBACEHIAUhBgNAIAdBAXQiEUEBciEKIAYiDCAHQQN0akEIaiEGAkAgDyARQQJqIgdMBEAgCiEHDAELIAYqAgAgBioCCF1FBEAgCiEHDAELIAZBCGohBgsgDCAGKgIAOAIAIAwgBigCBDYCBCAHIBBMDQALIA1BCGsiByAGRgRAIAYgEjgCACAGIA42AgQMAQsgBiAHKgIAOAIAIAYgDUEEayIKKAIANgIEIAcgEjgCACAKIA42AgAgBiAFa0EIaiIHQQlIDQAgBSAHQQN2QQJrQQF2IgdBA3RqIgoqAgAiEiAGKgIAIhNdRQ0AIAYoAgQhDANAAkAgBiASOAIAIAYgCiIGKAIENgIEIAdFDQAgBSAHQQFrQQF2IgdBA3RqIgoqAgAiEiATXQ0BCwsgBiAMNgIEIAYgEzgCAAsgCCANQQhrIgY2AiAMAQsLIAkgCCgCDEYNAwJAIAAoAowBIAAoAgwgCCgCLGxqIgIgACgChAFqLQACQQFxRQ0AIAggASACIAAoAoABaiAAKAKoASAAKAKkARELADgCCCAIQQxqIAhBCGogCEEsahDDASAAKAIkIAgoAhAiCyAIKAIMIgVrIgJBA3VPDQACQCACQQlJDQAgAkEDdiIMQQJrQQF2IQ0gBSgCBCEJIAUqAgAhEkEAIQcgBSEGA0AgB0EBdCIOQQFyIQIgBiIKIAdBA3RqQQhqIQYCQCAMIA5BAmoiB0wEQCACIQcMAQsgBioCACAGKgIIXUUEQCACIQcMAQsgBkEIaiEGCyAKIAYqAgA4AgAgCiAGKAIENgIEIAcgDUwNAAsgC0EIayICIAZGBEAgBiASOAIAIAYgCTYCBAwBCyAGIAIqAgA4AgAgBiALQQRrIgcoAgA2AgQgAiASOAIAIAcgCTYCACAGIAVrQQhqIgJBCUgNACAFIAJBA3ZBAmtBAXYiB0EDdGoiCioCACISIAYqAgAiE11FDQAgBigCBCECA0ACQCAGIBI4AgAgBiAKIgYoAgQ2AgQgB0UNACAFIAdBAWtBAXYiB0EDdGoiCioCACISIBNdDQELCyAGIAI2AgQgBiATOAIACyAIIAtBCGs2AhALIAAgAyAIQQxqIARBARDEASECDAIZIAgkACAIKAIMIgAEQCAIIAA2AhAgABDzAQsgCCgCHCIABEAgCCAANgIgIAAQ8wELCQALAAsgCEEwaiQADwsgCCgCDCEJCyAJBEAgCCAJNgIQIAkQ8wELIAgoAhwiBQRAIAggBTYCICAFEPMBCyAEQQFrIQQMAQsLAAsGQAZAQQgQ5AUhABgBIABBwB0QxAUhABkgCCQAIAAQ5QUJAAsgAEGskAJBARDmBQALmgYCBn8CfSMAIQggAigCACEGIAACfwJAIAEoAgQiBEUNAAJAIARpIgdBAk8EQCAGIQUgBCAGTQRAIAYgBHAhBQsgASgCACAFQQJ0aigCACICRQ0CIAdBAU0NAQNAIAIoAgAiAkUNAyAGIAIoAgQiB0cEQCAEIAdNBH8gByAEcAUgBwsgBUcNBAsgAigCCCAGRw0AC0EADAMLIAEoAgAgBEEBayAGcSIFQQJ0aigCACICRQ0BCyAEQQFrIQcDQCACKAIAIgJFDQEgBiACKAIEIglHIAcgCXEgBUdxDQEgAigCCCAGRw0AC0EADAELQQwQqAUhAiADKAIAIQMgAiAGNgIEIAIgAzYCCCACQQA2AgACQEEAIAQgASgCDEEBarMiCiABKgIQIgsgBLOUXhsNAEECIQUGQAJAAkAgBCAEQQFrcUEARyAEQQNJciAEQQF0ciIDAn8gCiALlY0iCkMAAIBPXSAKQwAAAABgcQRAIAqpDAELQQALIgcgAyAHSxsiA0EBRg0AIAMgA0EBa3FFBEAgAyEFDAELIAMQ+gEhBSABKAIEIQQLIAQgBU8EQCAEIAVNDQEgBEEDSSEHAn8gASgCDLMgASoCEJWNIgpDAACAT10gCkMAAAAAYHEEQCAKqQwBC0EACyEDIAUCfwJAIAcNACAEaUEBSw0AIANBAUEgIANBAWtna3QgA0ECSRsMAQsgAxD6AQsiAyADIAVJGyIFIARPDQELIAEgBRCvAQsZIAgkACACEPMBCQALIAEoAgQiBCAEQQFrIgNxRQRAIAMgBnEhBQwBCyAEIAZLBEAgBiEFDAELIAYgBHAhBQsCQAJAIAEoAgAgBUECdGoiBSgCACIDRQRAIAIgAUEIaiIDKAIANgIAIAEgAjYCCCAFIAM2AgAgAigCACIDRQ0CIAMoAgQhAwJAIAQgBEEBayIFcUUEQCADIAVxIQMMAQsgAyAESQ0AIAMgBHAhAwsgASgCACADQQJ0aiEDDAELIAIgAygCADYCAAsgAyACNgIACyABIAEoAgxBAWo2AgxBAQs6AAQgACACNgIAC90MAwt/An0BfiMAQTBrIgUkAAJAIAEoAgQiAyABKAIAIghrQQN1IAJJDQAgBUEANgIoIAVCADcCICAFQQA2AhwgBUIANwIUA0AGQCADIAhGBEBBACEEAkADQAJAIAUoAhQhBiAFKAIkIgMgBSgCICIHRg0AIAQgBmsiC0EDdSIMIAJPDQAgBygCACENIAcoAgQhCSAHIAMgAyAHa0EDdRCRASANQYCAgIB4c74hDyAFIAUoAiRBCGs2AiQgBiEDA0AgAyAERwRAIAAoAoABIgogACgCjAEiCCAAKAIMIgcgAygCBGxqaiAIIAcgCWxqIApqIAAoAqgBIAAoAqQBEQsAIQ4gA0EIaiEDIA4gD11FDQEMAwsLIAUoAhwgBEcEQCAEIAk2AgQgBCANNgIAIAUgBEEIaiIENgIYDAILIAxBAWoiB0GAgICAAk8EQBA3DAMLQf////8BIAtBAnYiAyAHIAMgB0sbIAtB+P///wdPGyIKBH8gCkGAgICAAk8EQBCbAQwECyAKQQN0EKgFBUEACyIIIAxBA3RqIgMgCTYCBCADIA02AgAgA0EIaiEHIAQgBkcEfwNAIANBCGsiAyAEQQhrIgQpAgA3AgAgBCAGRw0ACyAFKAIUBSAECyEGIAUgCCAKQQN0ajYCHCAFIAc2AhggBSADNgIUIAchBCAGRQ0BIAYQ8wEMAQsLIAVBCGpBBHIhACAGIQMDQCADIARGBEAgBgRAIAYQ8wELIAUoAiAiAEUNBiAFIAA2AiQgABDzAQwGCyAFIAMpAgAiEDcDCCAFIBCnQYCAgIB4czYCBCABIAVBBGogABDDASADQQhqIQMMAAsACwALIAUgCCoCAIw4AggCQAJAAkACQAJAAkAgBSgCJCIEIAUoAigiA0kEQCAEIAUqAgg4AgAgBCAIKAIENgIEIAUgBEEIaiIJNgIkDAELIAQgBSgCICIMa0EDdSIHQQFqIgpBgICAgAJPDQJB/////wEgAyAMayIGQQJ2IgMgCiADIApLGyAGQfj///8HTxsiA0GAgICAAk8NASADQQN0IgYQqAUiAyAHQQN0aiILIAUqAgg4AgAgCyAIKAIENgIEIAtBCGohCSAEIAxHBEADQCALQQhrIgsgBEEIayIEKQIANwIAIAQgDEcNAAsLIAUgAyAGajYCKCAFIAk2AiQgBSALNgIgIAxFDQAgDBDzASAFKAIkIQkLIAkgBSgCICIKayIDQQlIDQQCQCAKIANBA3ZBAmsiB0EBdiIIQQN0IgNqIgQqAgAiDiAJQQhrIgYqAgAiD10EQCAJQQRrKAIAIQsgAyAKaigCBCEDDAELIA4gD14NBSAKIAhBA3RqKAIEIgMgCUEEaygCACILTw0FCyAGIA44AgAgCUEEayADNgIAIAdBAkkNAgNAAkAgDyAKIAhBAWsiBkEBdiIIQQN0IgdqIgMqAgAiDl4EQCAHIApqKAIEIQkMAQsgDiAPXg0EIAcgCmooAgQiCSALTw0ECyAEIAk2AgQgBCAOOAIAIAMhBCAGQQFLDQALDAMLEJsBAAsQNwALIAQhAwsgAyALNgIEIAMgDzgCAAsZIAUkACAFKAIUIgAEQCAFIAA2AhggABDzAQsgBSgCICIABEAgBSAANgIkIAAQ8wELCQALAkAgASgCBCINIAEoAgAiCGsiA0EJSQ0AIANBA3YiC0ECa0EBdiEMIAgoAgQhCSAIKgIAIQ5BACEEIAghAwNAIARBAXQiCkEBciEGIAMiByAEQQN0akEIaiEDAkAgCyAKQQJqIgRMBEAgBiEEDAELIAMqAgAgAyoCCF1FBEAgBiEEDAELIANBCGohAwsgByADKgIAOAIAIAcgAygCBDYCBCAEIAxMDQALIA1BCGsiBiADRgRAIAMgDjgCACADIAk2AgQMAQsgAyAGKgIAOAIAIAMgDUEEayIEKAIANgIEIAYgDjgCACAEIAk2AgAgAyAIa0EIaiIEQQlIDQAgCCAEQQN2QQJrQQF2IgRBA3RqIgYqAgAiDyADKgIAIg5dRQ0AIAMoAgQhBwNAAkAgAyAPOAIAIAMgBiIDKAIENgIEIARFDQAgCCAEQQFrQQF2IgRBA3RqIgYqAgAiDyAOXQ0BCwsgAyAHNgIEIAMgDjgCAAsgASANQQhrIgM2AgQMAAsACyAFQTBqJAALjAIBBH8jACECAkAgACgCFCIDBEAgACgCBCIEIAAoAhAiAkEIdkH8//8HcWooAgAgAkH/B3FBAnRqKAIAIQEgACADQQFrNgIUIAAgAkEBaiICNgIQIAJBgBBJDQEgBCgCABDzASAAIAAoAgRBBGo2AgQgACAAKAIQQYAIazYCEAwBCwZAQQwQqAUiASAAKAIwIgA2AgggAUH//wM7AQAGQEF/IABBAXQgAEEASBsQqAUhABkgAiQAIAEQ8wEJAAsZIAIkAAkACyABIAA2AgQLIAEgAS8BAEEBaiIAOwEAIAAgAEH//wNxRwRAIAEoAgRBACABKAIIQQF0ENQBGiABIAEvAQBBAWo7AQALIAELshABBX8jAEGQAmsiBCQABkACQAZAIARB0ABqIAEQsAEhBRgCAkAgBSgCSEUEQAZABkBBCBDkBSEAGAUgAEHeJBDEBSEADAIZIAQkAAZAIAAQ5QUYBQkACwALIAVCAEECEKYCIARBQGsgBRClAiAFQgBBABCmAiAFIABBhAFqQQQQpAIgBSAAQQRqQQQQpAIgBSAAQQhqQQQQpAIgACAAKAIEIAMgACgCCCADSxsiBjYCBCAFIABBDGpBBBCkAiAFIABBiAFqQQQQpAIgBSAAQYABakEEEKQCIAUgAEFAa0EEEKQCIAUgAEH4AGpBBBCkAiAFIABBHGpBBBCkAiAFIABBIGpBBBCkAiAFIABBGGpBBBCkAiAFIABBMGpBCBCkAiAFIABBJGpBBBCkAiAAIAIgAigCACgCABEBADYCoAEgACACIAIoAgAoAgQRAQA2AqQBIAAgAiACKAIAKAIIEQEANgKoASAEQTBqIAUQpQIgBSAAKAIMIAAoAghsrUEBEKYCQQAhAgJAA0AgACgCCCACTQRAIARBIGogBRClAiAEKQMoIAQpA0hSBEAGQAZAQQgQ5AUhABgIIABB2SYQxAUhAAwEGSAEJAAGQCAAEOUFGAgJAAsACyAFIAUoAgBBDGsoAgBqQQAQ3gIgBSAEKQM4QQAQpgIgACAAKAIMIAZsEPIBIgE2AowBAkAgAUUEQAZABkBBCBDkBSEAGAkgAEGDOBDEBSEADAIZIAQkAAZAIAAQ5QUYCQkACwALIAUgASAAKAIMIAAoAghsEKQCIAAgACgCHEECdEEEajYCECAAIAAoAiBBAnRBBGo2AnxBACEBIARBADYCKCAEQgA3AiAgBEEAOgAUIAQgBEEgajYCEEEAIQIgBgRABkAgBkGr1arVAE8EQBA3DAgLIAZBGGwiARCoBSECGSAEJAAgBEEQahCzAQkACyACQQAgAUEYayIDIANBGHBrQRhqIgMQ1AEiCCADaiEHIAEgCGohAQsgBCAAKAJsIgM2AiAgACACNgJsIAQgACgCcCICNgIkIAAgBzYCcCAEIAAoAnQ2AiggACABNgJ0IAMEQCADIgEgAkcEQANAIAJBGGsiAiADRw0ACyAEKAIgIQELIAQgAzYCJCABEPMBCyAEQQA2AiggBEIANwIgIARBADoAFCAEIARBIGo2AhAGQEGAgOAAEKgFIQEZIAQkACAEQRBqELMBCQALIAFBAEGAgOAAENQBIQEgBCAAKAJIIgM2AiAgACABNgJIIAQgACgCTCICNgIkIAAgAUGAgOAAaiIBNgJMIAQgACgCUDYCKCAAIAE2AlAgAwRAIAMiASACRwRAA0AgAkEYayICIANHDQALIAQoAiAhAQsgBCADNgIkIAEQ8wELQTQQqAUhAQZAIAEgBhC0ASEBGSAEJAAgARDzAQkACyAAIAE2AkQgACAGQQJ0IgMQ8gEiATYCkAECQCABRQRABkAGQEEIEOQFIQAYCiAAQccREMQFIQAMAhkgBCQABkAgABDlBRgKCQALAAtBACEBIARBADYCKCAEQgA3AiBBACEHQQAhAiAGBEAGQCAGQYCAgIAETwRAEDcMCQsgAxCoBSECGSAEJAAgBCgCICIABEAgBCAANgIkIAAQ8wELCQALIAJBACADENQBIgEgA2ohByABIAZBAnRqIQELIAAoApQBIgMEQCAAIAM2ApgBIAMQ8wEgAEEANgKcASAAQgA3ApQBCyAAIAI2ApQBIAAgATYCnAEgACAHNgKYASAAQQo2AiggAEQAAAAAAADwPyAAKwMwozkDOCAAQcQBaiEDQQAhAgNAIAAoAgggAk0EQCAAQYQCaiEBQQAhAgNAIAAoAgggAk0EQCAFQQhqIgAQvQJFBEAgBSAFKAIAQQxrKAIAaiIBIAEoAhBBBHIQ3gILIAVBrJYBKAIAIgE2AgAgBSABQQxrKAIAakG4lgEoAgA2AgAgABC8AhogBUHsAGoQigIgBEGQAmokAA8LAkAgACgChAEgACgCjAEgACgCDCACbGpqLQACQQFxRQ0AIAAgACgCFEEBajYCFCAALQDoAUUNACAEIAI2AhAgBEEgaiABIARBEGoiAyADEMYBCyACQQFqIQIMAAsACyAEIAAoAogBIAAoAowBIAAoAgwgAmxqaigAADYCDCAEIARBDGoiATYCECAEQSBqIgYgAyABIARBEGogBEGPAmoQjAEgBCgCICACNgIMIAUgBkEEEKQCIAQoAiAiBkUEQCACQQJ0IgEgACgClAFqQQA2AgAgACgCkAEgAWpBADYCACACQQFqIQIMAQsgAkECdCIBIAAoApQBaiAGIAAoAhBuNgIAIAQoAiAiBhDyASEHIAAoApABIAFqIAc2AgACQCAAKAKQASABaigCACIBRQRABkAGQEEIEOQFIQAYDCAAQcUMEMQFIQAMAhkgBCQABkAgABDlBRgMCQALAAsgBSABIAYQpAIgAkEBaiECDAELCyAAQayQAkEBEOYFDAYLIABBrJACQQEQ5gUMBQsgAEGskAJBARDmBQwECyAEQSBqIAUQpQICQAJAIAQpAyhCAFkEQCAEQRBqIAUQpQIgBCkDGCAEKQNIUw0BCwZABkBBCBDkBSEAGAggAEHZJhDEBSEADAIZIAQkAAZAIAAQ5QUYCAkACwALIAUgBEEgakEEEKQCIAQoAiAiAQRAIAUgAa1BARCmAgsgAkEBaiECDAELCyAAQayQAkEBEOYFDAILIABBrJACQQEQ5gUMAQsgAEGskAJBARDmBQsZIAQkACAFELEBGgkACwAL5QoCCH8BfCMAQSBrIgMkAAZABkAgASgCACECQYwfEAwhBRgBIAIgBRARIQIZIAMkAAZAIAUQABkgAyQAEPAFAAsJAAsGQCAFEAAZIAMkABDwBQALBkAgAkHsigIgA0EQahASIQoZIAMkAAZAIAIQABkgAyQAEPAFAAsJAAsGQCADKAIQEBMZIAMkABDwBQALAn8gCkQAAAAAAADwQWMgCkQAAAAAAAAAAGZxBEAgCqsMAQtBAAshCQZAIAIQABkgAyQAEPAFAAsgAEEANgIIIABCADcCAAZAAkAgACgCCCAAKAIAIgJrQQxtIAlPDQACQAJAIAlB1qrVqgFJBEAgACgCBCEHIAlBDGwiBhCoBSIFIAZqIQYgBSAHIAJrQQxtQQxsaiEFIAIgB0YNASAFIQQDQCAEQQxrIgQgB0EMayIHKAIANgIAIAQgBygCBDYCBCAEIAcoAgg2AgggB0EANgIIIAdCADcCACACIAdHDQALIAAgBjYCCCAAKAIEIQIgACAFNgIEIAAoAgAhBiAAIAQ2AgAgAiAGRg0CA0AgAkEMayIFKAIAIgQEQCACQQhrIAQ2AgAgBBDzAQsgBSICIAZHDQALIAYhAgwCCxA3AAsgACAGNgIIIAAgBTYCBCAAIAU2AgALIAJFDQAgAhDzAQtBACEHA0AgByAJTwRAIANBIGokAA8LIAEoAgAhAiADIAc2AhBB7IoCIANBEGoQCSEFBkAgAiAFEBEhAhkgAyQABkAgBRAAGSADJAAQ8AUACwkACyADIAI2AgwGQCAFEAAZIAMkABDwBQALAkAGQCADQRBqIQUjAEEQayIEJAACfyADKAIMQZjUACAEQQhqEBIiCkQAAAAAAADwQWMgCkQAAAAAAAAAAGZxBEAgCqsMAQtBAAshAiAEKAIIIQYgBCACNgIMBkAgBSAEQQxqEJkBGSAEJAAGQCAEKAIMEAAZIAQkABDwBQALBkAgBhATGSAEJAAQ8AUACwkACwZAIAQoAgwQABkgBCQAEPAFAAsGQCAGEBMZIAQkABDwBQALIARBEGokACAAKAIEIgIgACgCCEkEQCACQQA2AgggAkIANwIAIAIgAygCEDYCACACIAMoAhQ2AgQgAiADKAIYNgIIIANBADYCGCADQgA3AhAgACACQQxqNgIEDAILBkACQAJAAkAgACgCBCAAKAIAIgJrQQxtIgZBAWoiBEHWqtWqAUkEQEHVqtWqASAAKAIIIAJrQQxtIgVBAXQiAiAEIAIgBEsbIAVBqtWq1QBPGyICQdaq1aoBTw0BIAJBDGwiBRCoBSICIAZBDGxqIgggAygCEDYCACAIIAMoAhQ2AgQgCCADKAIYNgIIIANBADYCGCADQgA3AhAgAiAFaiEEIAhBDGohBiAAKAIEIgIgACgCACIFRg0CA0AgCEEMayIIIAJBDGsiAigCADYCACAIIAIoAgQ2AgQgCCACKAIINgIIIAJBADYCCCACQgA3AgAgAiAFRw0ACyAAIAQ2AgggACgCBCECIAAgBjYCBCAAKAIAIQYgACAINgIAIAIgBkYNAwNAIAJBDGsiBSgCACIEBEAgAkEIayAENgIAIAQQ8wELIAUiAiAGRw0ACyAGIQIMAwsQNwALEJsBAAsgACAENgIIIAAgBjYCBCAAIAg2AgALIAIEQCACEPMBCxkgAyQAIAMoAhAiAQRAIAMgATYCFCABEPMBCwkACxkgAyQABkAgAygCDBAAGSADJAAQ8AUACwkACyADKAIQIgJFDQAgAyACNgIUIAIQ8wELBkAgAygCDBAAGSADJAAQ8AUACyAHQQFqIQcMAAsAGSADJAAgABDMAQkACwAL7gUCC38BfCMAQRBrIgIkAAZABkAgASgCACEDQYwfEAwhBBgBIAMgBBARIQMZIAIkAAZAIAQQABkgAiQAEPAFAAsJAAsGQCAEEAAZIAIkABDwBQALBkAgA0HsigIgAkEIahASIQ0ZIAIkAAZAIAMQABkgAiQAEPAFAAsJAAsGQCACKAIIEBMZIAIkABDwBQALAn8gDUQAAAAAAADwQWMgDUQAAAAAAAAAAGZxBEAgDasMAQtBAAshCAZAIAMQABkgAiQAEPAFAAsgAEEANgIIIABCADcCAEEAIQQGQAJAIAgEQCAIQYCAgIAETwRAEDcMAgsgACAIQQJ0IgMQqAUiBDYCBCAAIAQ2AgAgACADIARqIgY2AggLIAQhAwNAIAggCU0EQCACQRBqJAAPCyABKAIAIQcgAiAJNgIIQeyKAiACQQhqEAkhBQZAIAcgBRARIQcZIAIkAAZAIAUQABkgAiQAEPAFAAsJAAsGQCAFEAAZIAIkABDwBQALAkAGQAJ/IAdB1IoCIAJBCGoQEiENBkAgAigCCBATGSACJAAQ8AUACyADIAZPIQoCfyANRAAAAAAAAPBBYyANRAAAAAAAAAAAZnEEQCANqwwBC0EACyEFIApFBEAgAyAFNgIAIAAgA0EEaiIDNgIEDAMLIAMgBGsiCkECdSILQQFqIgNBgICAgARPBEAQNwwFC0EAQf////8DIAYgBGsiBkEBdiIMIAMgAyAMSRsgBkH8////B08bIgNFDQAaIANBgICAgARPBEAQmwEMBQsgA0ECdBCoBQshBhkgAiQABkAgBxAAGSACJAAQ8AUACwkACyAGIAtBAnRqIgsgBTYCACAAIAYgBCAKENMBIgUgA0ECdGoiBjYCCCAAIAtBBGoiAzYCBCAAIAU2AgAgBARAIAQQ8wELIAUhBAsGQCAHEAAZIAIkABDwBQALIAlBAWohCQwACwALGSACJAAgACgCACIBBEAgACABNgIEIAEQ8wELCQALAAtcAQR/IAAoAgAiAgRAIAIhASACIAAoAgQiA0cEQANAIANBDGsiASgCACIEBEAgA0EIayAENgIAIAQQ8wELIAEiAyACRw0ACyAAKAIAIQELIAAgAjYCBCABEPMBCwulAQECfyMAQRBrIgEkACAAKAIEIQIgASAAKAIAIgA2AgwgASACIABrQQJ1NgIIBkAGQEGE4gAgAUEIahAJIQAYASAAEBAhAhkgASQABkAgABAAGSABJAAQ8AUACwkACwZAIAAQABkgASQAEPAFAAsGQCACEAIZIAEkAAZAIAIQABkgASQAEPAFAAsJAAsGQCACEAAZIAEkABDwBQALIAFBEGokACACC9QBAQJ/IwBBEGsiAiQAIAIgATYCBAJAAkAgASAAKAIISQRAIAAoAoQBIAAoAowBIAAoAgwgAigCBGxqaiIBLQACIgNBAXENASABIANBAXI6AAIgACAAKAIUQQFqNgIUIAAtAOgBRQ0CBkAgAkEIaiAAQYQCaiACQQRqIgAgABDGAQwDGSACJAAJAAsAC0H4DkGxIUGTBkHLHhAVAAsGQAZAQQgQ5AUhABgCIABBmScQxAUhABkgAiQAIAAQ5QUJAAsgAEGskAJBARDmBQALIAJBEGokAAsnAQJ/IAAoAgQiABDjAUEBaiIBEPIBIgIEfyACIAAgARDSAQVBAAsLJAEBf0GU5AIoAgAiAARAA0AgACgCABEMACAAKAIEIgANAAsLC40EAEHoiQJBiyYQFkGAigJBoh1BAUEBQQAQF0GMigJB9xhBAUGAf0H/ABAYQaSKAkHwGEEBQYB/Qf8AEBhBmIoCQe4YQQFBAEH/ARAYQbCKAkHSDkECQYCAfkH//wEQGEG8igJByQ5BAkEAQf//AxAYQciKAkHdD0EEQYCAgIB4Qf////8HEBhB1IoCQdQPQQRBAEF/EBhB4IoCQesiQQRBgICAgHhB/////wcQGEHsigJB4iJBBEEAQX8QGEH4igJBnRFCgICAgICAgICAf0L///////////8AEPcHQYSLAkGcEUIAQn8Q9wdBkIsCQeMQQQQQGUGciwJBgyVBCBAZQcDcAEGWIxAaQYzlAEHnMxAaQdTlAEEEQfwiEBtBoOYAQQJBoiMQG0Hs5gBBBEGxIxAbQZjUAEGqHhAcQZTnAEEAQe0yEB1BvOcAQQBBiDQQHUHk5wBBAUHAMxAdQYzoAEECQbAvEB1BtOgAQQNBzy8QHUHc6ABBBEH3LxAdQYTiAEEFQZQwEB1BhOkAQQRBrTQQHUGs6QBBBUHLNBAdQbznAEEAQfowEB1B5OcAQQFB2TAQHUGM6ABBAkG8MRAdQbToAEEDQZoxEB1B3OgAQQRBwjIQHUGE4gBBBUGgMhAdQdTpAEEIQf8xEB1B/OkAQQlB3TEQHUHA1ABBBkG6MBAdQaTqAEEHQfI0EB0LgAQBA38gAkGABE8EQCAAIAEgAhAeIAAPCyAAIAJqIQMCQCAAIAFzQQNxRQRAAkAgAEEDcUUEQCAAIQIMAQsgAkUEQCAAIQIMAQsgACECA0AgAiABLQAAOgAAIAFBAWohASACQQFqIgJBA3FFDQEgAiADSQ0ACwsCQCADQXxxIgRBwABJDQAgAiAEQUBqIgVLDQADQCACIAEoAgA2AgAgAiABKAIENgIEIAIgASgCCDYCCCACIAEoAgw2AgwgAiABKAIQNgIQIAIgASgCFDYCFCACIAEoAhg2AhggAiABKAIcNgIcIAIgASgCIDYCICACIAEoAiQ2AiQgAiABKAIoNgIoIAIgASgCLDYCLCACIAEoAjA2AjAgAiABKAI0NgI0IAIgASgCODYCOCACIAEoAjw2AjwgAUFAayEBIAJBQGsiAiAFTQ0ACwsgAiAETw0BA0AgAiABKAIANgIAIAFBBGohASACQQRqIgIgBEkNAAsMAQsgA0EESQRAIAAhAgwBCyAAIANBBGsiBEsEQCAAIQIMAQsgACECA0AgAiABLQAAOgAAIAIgAS0AAToAASACIAEtAAI6AAIgAiABLQADOgADIAFBBGohASACQQRqIgIgBE0NAAsLIAIgA0kEQANAIAIgAS0AADoAACABQQFqIQEgAkEBaiICIANHDQALCyAAC+kCAQJ/AkAgACABRg0AIAEgACACaiIEa0EAIAJBAXRrTQRAIAAgASACENIBDwsgACABc0EDcSEDAkACQCAAIAFJBEAgAwRAIAAhAwwDCyAAQQNxRQRAIAAhAwwCCyAAIQMDQCACRQ0EIAMgAS0AADoAACABQQFqIQEgAkEBayECIANBAWoiA0EDcQ0ACwwBCwJAIAMNACAEQQNxBEADQCACRQ0FIAAgAkEBayICaiIDIAEgAmotAAA6AAAgA0EDcQ0ACwsgAkEDTQ0AA0AgACACQQRrIgJqIAEgAmooAgA2AgAgAkEDSw0ACwsgAkUNAgNAIAAgAkEBayICaiABIAJqLQAAOgAAIAINAAsMAgsgAkEDTQ0AA0AgAyABKAIANgIAIAFBBGohASADQQRqIQMgAkEEayICQQNLDQALCyACRQ0AA0AgAyABLQAAOgAAIANBAWohAyABQQFqIQEgAkEBayICDQALCyAAC/ICAgJ/AX4CQCACRQ0AIAAgAToAACAAIAJqIgNBAWsgAToAACACQQNJDQAgACABOgACIAAgAToAASADQQNrIAE6AAAgA0ECayABOgAAIAJBB0kNACAAIAE6AAMgA0EEayABOgAAIAJBCUkNACAAQQAgAGtBA3EiBGoiAyABQf8BcUGBgoQIbCIBNgIAIAMgAiAEa0F8cSIEaiICQQRrIAE2AgAgBEEJSQ0AIAMgATYCCCADIAE2AgQgAkEIayABNgIAIAJBDGsgATYCACAEQRlJDQAgAyABNgIYIAMgATYCFCADIAE2AhAgAyABNgIMIAJBEGsgATYCACACQRRrIAE2AgAgAkEYayABNgIAIAJBHGsgATYCACAEIANBBHFBGHIiBGsiAkEgSQ0AIAGtQoGAgIAQfiEFIAMgBGohAQNAIAEgBTcDGCABIAU3AxAgASAFNwMIIAEgBTcDACABQSBqIQEgAkEgayICQR9LDQALCyAAC+cEAwF/BnwCfiAAvSIIQjCIpyEBIAhCgICAgICAgPc/fUL//////5/CAVgEQCAIQoCAgICAgID4P1EEQEQAAAAAAAAAAA8LIABEAAAAAAAA8L+gIgAgACAARAAAAAAAAKBBoiICoCACoSICIAKiQejqACsDACIFoiIGoCIHIAAgACAAoiIDoiIEIAQgBCAEQbjrACsDAKIgA0Gw6wArAwCiIABBqOsAKwMAokGg6wArAwCgoKCiIANBmOsAKwMAoiAAQZDrACsDAKJBiOsAKwMAoKCgoiADQYDrACsDAKIgAEH46gArAwCiQfDqACsDAKCgoKIgACACoSAFoiAAIAKgoiAGIAAgB6GgoKCgDwsCQCABQfD/AWtBn4B+TQRAIAhC////////////AINQBEAjAEEQayIBRAAAAAAAAPC/OQMIIAErAwhEAAAAAAAAAACjDwsgCEKAgICAgICA+P8AUQ0BIAFB8P8BcUHw/wFHIAFB//8BTXFFBEAgACAAoSIAIACjDwsgAEQAAAAAAAAwQ6K9QoCAgICAgICgA30hCAsgCEKAgICAgICA8z99IglCNIentyIDQbDqACsDAKIgCUItiKdB/wBxQQR0IgFByOsAaisDAKAiBCABQcDrAGorAwAgCCAJQoCAgICAgIB4g32/IAFBwPsAaisDAKEgAUHI+wBqKwMAoaIiAKAiBSAAIAAgAKIiAqIgAiAAQeDqACsDAKJB2OoAKwMAoKIgAEHQ6gArAwCiQcjqACsDAKCgoiACQcDqACsDAKIgA0G46gArAwCiIAAgBCAFoaCgoKCgIQALIAAL4wEBAn8gAkEARyEDAkACQAJAIABBA3FFDQAgAkUNACABQf8BcSEEA0AgAC0AACAERg0CIAJBAWsiAkEARyEDIABBAWoiAEEDcUUNASACDQALCyADRQ0BAkAgAC0AACABQf8BcUYNACACQQRJDQAgAUH/AXFBgYKECGwhAwNAIAAoAgAgA3MiBEF/cyAEQYGChAhrcUGAgYKEeHENAiAAQQRqIQAgAkEEayICQQNLDQALCyACRQ0BCyABQf8BcSEBA0AgASAALQAARgRAIAAPCyAAQQFqIQAgAkEBayICDQALC0EAC4EBAQJ/AkACQCACQQRPBEAgACABckEDcQ0BA0AgACgCACABKAIARw0CIAFBBGohASAAQQRqIQAgAkEEayICQQNLDQALCyACRQ0BCwNAIAAtAAAiAyABLQAAIgRGBEAgAUEBaiEBIABBAWohACACQQFrIgINAQwCCwsgAyAEaw8LQQALKwEBfyMAQRBrIgIkACACIAE2AgxBoNoCIAAgAUEAQQAQ5gEaIAJBEGokAAsEAEEBCwQAQQALBABBAAtZAQF/IAAgACgCSCIBQQFrIAFyNgJIIAAoAgAiAUEIcQRAIAAgAUEgcjYCAEF/DwsgAEIANwIEIAAgACgCLCIBNgIcIAAgATYCFCAAIAEgACgCMGo2AhBBAAvCAQEDfwJAIAEgAigCECIDBH8gAwUgAhDcAQ0BIAIoAhALIAIoAhQiBWtLBEAgAiAAIAEgAigCJBEEAA8LAkAgAigCUEEASARAQQAhAwwBCyABIQQDQCAEIgNFBEBBACEDDAILIAAgA0EBayIEai0AAEEKRw0ACyACIAAgAyACKAIkEQQAIgQgA0kNASAAIANqIQAgASADayEBIAIoAhQhBQsgBSAAIAEQ0gEaIAIgAigCFCABajYCFCABIANqIQQLIAQLQgEBfyABIAJsIQQgBAJ/IAMoAkxBAEgEQCAAIAQgAxDdAQwBCyAAIAQgAxDdAQsiAEYEQCACQQAgARsPCyAAIAFuC30BAn8jAEEQayIBJAAgAUEKOgAPAkACQCAAKAIQIgIEfyACBSAAENwBDQIgACgCEAsgACgCFCICRg0AIAAoAlBBCkYNACAAIAJBAWo2AhQgAkEKOgAADAELIAAgAUEPakEBIAAoAiQRBABBAUcNACABLQAPGgsgAUEQaiQAC2wBAX9B7NoCKAIAGgJAQX9BACAAQQEgABDjASIAQaDaAhDeASAARxtBAEgNAAJAQfDaAigCAEEKRg0AQbTaAigCACIAQbDaAigCAEYNAEG02gIgAEEBajYCACAAQQo6AAAMAQtBoNoCEN8BCwv2AgEHfyMAQSBrIgMkACADIAAoAhwiBDYCECAAKAIUIQUgAyACNgIcIAMgATYCGCADIAUgBGsiATYCFCABIAJqIQVBAiEHAn8CQAJAAkAgACgCPCADQRBqIgFBAiADQQxqEB8iBAR/QbjsAiAENgIAQX8FQQALBEAgASEEDAELA0AgBSADKAIMIgZGDQIgBkEASARAIAEhBAwECyABIAYgASgCBCIISyIJQQN0aiIEIAYgCEEAIAkbayIIIAQoAgBqNgIAIAFBDEEEIAkbaiIBIAEoAgAgCGs2AgAgBSAGayEFIAAoAjwgBCIBIAcgCWsiByADQQxqEB8iBgR/QbjsAiAGNgIAQX8FQQALRQ0ACwsgBUF/Rw0BCyAAIAAoAiwiATYCHCAAIAE2AhQgACABIAAoAjBqNgIQIAIMAQsgAEEANgIcIABCADcDECAAIAAoAgBBIHI2AgBBACAHQQJGDQAaIAIgBCgCBGsLIQAgA0EgaiQAIAALBABCAAtpAQN/AkAgACIBQQNxBEADQCABLQAARQ0CIAFBAWoiAUEDcQ0ACwsDQCABIgJBBGohASACKAIAIgNBf3MgA0GBgoQIa3FBgIGChHhxRQ0ACwNAIAIiAUEBaiECIAEtAAANAAsLIAEgAGsLBgBBuOwCC38CAX8BfiAAvSIDQjSIp0H/D3EiAkH/D0cEfCACRQRAIAEgAEQAAAAAAAAAAGEEf0EABSAARAAAAAAAAPBDoiABEOUBIQAgASgCAEFAags2AgAgAA8LIAEgAkH+B2s2AgAgA0L/////////h4B/g0KAgICAgICA8D+EvwUgAAsL1gIBBH8jAEHQAWsiBSQAIAUgAjYCzAEgBUGgAWoiAkEAQSgQ1AEaIAUgBSgCzAE2AsgBAkBBACABIAVByAFqIAVB0ABqIAIgAyAEEOcBQQBIBEBBfyEEDAELIAAoAkxBAE4hBiAAKAIAIQcgACgCSEEATARAIAAgB0FfcTYCAAsCQAJAAkAgACgCMEUEQCAAQdAANgIwIABBADYCHCAAQgA3AxAgACgCLCEIIAAgBTYCLAwBCyAAKAIQDQELQX8hAiAAENwBDQELIAAgASAFQcgBaiAFQdAAaiAFQaABaiADIAQQ5wEhAgsgCARAIABBAEEAIAAoAiQRBAAaIABBADYCMCAAIAg2AiwgAEEANgIcIAAoAhQhASAAQgA3AxAgAkF/IAEbIQILIAAgACgCACIAIAdBIHFyNgIAQX8gAiAAQSBxGyEEIAZFDQALIAVB0AFqJAAgBAv2EgISfwF+IwBB0ABrIggkACAIIAE2AkwgCEE3aiEXIAhBOGohEgJAAkACQAJAA0AgASEMIAcgDkH/////B3NKDQEgByAOaiEOAkACQAJAIAwiBy0AACIJBEADQAJAAkAgCUH/AXEiAUUEQCAHIQEMAQsgAUElRw0BIAchCQNAIAktAAFBJUcEQCAJIQEMAgsgB0EBaiEHIAktAAIhCyAJQQJqIgEhCSALQSVGDQALCyAHIAxrIgcgDkH/////B3MiGEoNByAABEAgACAMIAcQ6AELIAcNBiAIIAE2AkwgAUEBaiEHQX8hDwJAIAEsAAFBMGtBCk8NACABLQACQSRHDQAgAUEDaiEHIAEsAAFBMGshD0EBIRMLIAggBzYCTEEAIQ0CQCAHLAAAIglBIGsiAUEfSwRAIAchCwwBCyAHIQtBASABdCIBQYnRBHFFDQADQCAIIAdBAWoiCzYCTCABIA1yIQ0gBywAASIJQSBrIgFBIE8NASALIQdBASABdCIBQYnRBHENAAsLAkAgCUEqRgRAAn8CQCALLAABQTBrQQpPDQAgCy0AAkEkRw0AIAssAAFBAnQgBGpBwAFrQQo2AgAgC0EDaiEJQQEhEyALLAABQQN0IANqQYADaygCAAwBCyATDQYgC0EBaiEJIABFBEAgCCAJNgJMQQAhE0EAIRAMAwsgAiACKAIAIgFBBGo2AgBBACETIAEoAgALIRAgCCAJNgJMIBBBAE4NAUEAIBBrIRAgDUGAwAByIQ0MAQsgCEHMAGoQ6QEiEEEASA0IIAgoAkwhCQtBACEHQX8hCgJAIAktAABBLkcEQCAJIQFBACEUDAELIAktAAFBKkYEQAJ/AkAgCSwAAkEwa0EKTw0AIAktAANBJEcNACAJLAACQQJ0IARqQcABa0EKNgIAIAlBBGohASAJLAACQQN0IANqQYADaygCAAwBCyATDQYgCUECaiEBQQAgAEUNABogAiACKAIAIgtBBGo2AgAgCygCAAshCiAIIAE2AkwgCkF/c0EfdiEUDAELIAggCUEBajYCTEEBIRQgCEHMAGoQ6QEhCiAIKAJMIQELA0AgByEVQRwhCyABIhEsAAAiB0H7AGtBRkkNCSARQQFqIQEgByAVQTpsakH/igFqLQAAIgdBAWtBCEkNAAsgCCABNgJMAkACQCAHQRtHBEAgB0UNCyAPQQBOBEAgBCAPQQJ0aiAHNgIAIAggAyAPQQN0aikDADcDQAwCCyAARQ0IIAhBQGsgByACIAYQ6gEMAgsgD0EATg0KC0EAIQcgAEUNBwsgDUH//3txIgkgDSANQYDAAHEbIQ1BACEPQd0LIRYgEiELAkACQAJAAn8CQAJAAkACQAJ/AkACQAJAAkACQAJAAkAgESwAACIHQV9xIAcgB0EPcUEDRhsgByAVGyIHQdgAaw4hBBQUFBQUFBQUDhQPBg4ODhQGFBQUFAIFAxQUCRQBFBQEAAsCQCAHQcEAaw4HDhQLFA4ODgALIAdB0wBGDQkMEwsgCCkDQCEZQd0LDAULQQAhBwJAAkACQAJAAkACQAJAIBVB/wFxDggAAQIDBBoFBhoLIAgoAkAgDjYCAAwZCyAIKAJAIA42AgAMGAsgCCgCQCAOrDcDAAwXCyAIKAJAIA47AQAMFgsgCCgCQCAOOgAADBULIAgoAkAgDjYCAAwUCyAIKAJAIA6sNwMADBMLQQggCiAKQQhNGyEKIA1BCHIhDUH4ACEHCyASIQwgB0EgcSERIAgpA0AiGUIAUgRAA0AgDEEBayIMIBmnQQ9xQZCPAWotAAAgEXI6AAAgGUIPViEJIBlCBIghGSAJDQALCyAIKQNAUA0DIA1BCHFFDQMgB0EEdkHdC2ohFkECIQ8MAwsgEiEHIAgpA0AiGUIAUgRAA0AgB0EBayIHIBmnQQdxQTByOgAAIBlCB1YhDCAZQgOIIRkgDA0ACwsgByEMIA1BCHFFDQIgCiASIAxrIgdBAWogByAKSBshCgwCCyAIKQNAIhlCAFMEQCAIQgAgGX0iGTcDQEEBIQ9B3QsMAQsgDUGAEHEEQEEBIQ9B3gsMAQtB3wtB3QsgDUEBcSIPGwshFiAZIBIQ6wEhDAsgCkEASCAUcQ0OIA1B//97cSANIBQbIQ0CQCAIKQNAIhlCAFINACAKDQAgEiEMQQAhCgwMCyAKIBlQIBIgDGtqIgcgByAKSBshCgwLCyAIKAJAIgdByD0gBxsiDEEAQf////8HIAogCkH/////B08bIgsQ1gEiByAMayALIAcbIgcgDGohCyAKQQBOBEAgCSENIAchCgwLCyAJIQ0gByEKIAstAAANDQwKCyAKBEAgCCgCQAwCC0EAIQcgAEEgIBBBACANEOwBDAILIAhBADYCDCAIIAgpA0A+AgggCCAIQQhqIgc2AkBBfyEKIAcLIQlBACEHAkADQCAJKAIAIgxFDQECQCAIQQRqIAwQ8QEiC0EASCIMDQAgCyAKIAdrSw0AIAlBBGohCSAKIAcgC2oiB0sNAQwCCwsgDA0NC0E9IQsgB0EASA0LIABBICAQIAcgDRDsASAHRQRAQQAhBwwBC0EAIQsgCCgCQCEJA0AgCSgCACIMRQ0BIAhBBGogDBDxASIMIAtqIgsgB0sNASAAIAhBBGogDBDoASAJQQRqIQkgByALSw0ACwsgAEEgIBAgByANQYDAAHMQ7AEgECAHIAcgEEgbIQcMCAsgCkEASCAUcQ0IQT0hCyAAIAgrA0AgECAKIA0gByAFER0AIgdBAE4NBwwJCyAIIAgpA0A8ADdBASEKIBchDCAJIQ0MBAsgBy0AASEJIAdBAWohBwwACwALIAANByATRQ0CQQEhBwNAIAQgB0ECdGooAgAiAARAIAMgB0EDdGogACACIAYQ6gFBASEOIAdBAWoiB0EKRw0BDAkLC0EBIQ4gB0EKTw0HA0AgBCAHQQJ0aigCAA0BIAdBAWoiB0EKRw0ACwwHC0EcIQsMBAsgCiALIAxrIhEgCiARShsiCSAPQf////8Hc0oNAkE9IQsgECAJIA9qIgogCiAQSBsiByAYSg0DIABBICAHIAogDRDsASAAIBYgDxDoASAAQTAgByAKIA1BgIAEcxDsASAAQTAgCSARQQAQ7AEgACAMIBEQ6AEgAEEgIAcgCiANQYDAAHMQ7AEMAQsLQQAhDgwDC0E9IQsLQbjsAiALNgIAC0F/IQ4LIAhB0ABqJAAgDgsYACAALQAAQSBxRQRAIAEgAiAAEN0BGgsLcgEDfyAAKAIALAAAQTBrQQpPBEBBAA8LA0AgACgCACEDQX8hASACQcyZs+YATQRAQX8gAywAAEEwayIBIAJBCmwiAmogASACQf////8Hc0obIQELIAAgA0EBajYCACABIQIgAywAAUEwa0EKSQ0ACyACC8QCAAJAAkACQAJAAkACQAJAAkACQAJAAkACQAJAIAFBCWsOEgAKCwwKCwIDBAUMCwwMCgsHCAkLIAIgAigCACIBQQRqNgIAIAAgASgCADYCAA8LAAsgAiACKAIAIgFBBGo2AgAgACABMgEANwMADwsgAiACKAIAIgFBBGo2AgAgACABMwEANwMADwsgAiACKAIAIgFBBGo2AgAgACABMAAANwMADwsgAiACKAIAIgFBBGo2AgAgACABMQAANwMADwsACyACIAIoAgBBB2pBeHEiAUEIajYCACAAIAErAwA5AwAPCyAAIAIgAxEAAAsPCyACIAIoAgAiAUEEajYCACAAIAE0AgA3AwAPCyACIAIoAgAiAUEEajYCACAAIAE1AgA3AwAPCyACIAIoAgBBB2pBeHEiAUEIajYCACAAIAEpAwA3AwALgwECA38BfgJAIABCgICAgBBUBEAgACEFDAELA0AgAUEBayIBIAAgAEIKgCIFQgp+fadBMHI6AAAgAEL/////nwFWIQIgBSEAIAINAAsLIAWnIgIEQANAIAFBAWsiASACIAJBCm4iA0EKbGtBMHI6AAAgAkEJSyEEIAMhAiAEDQALCyABC3IBAX8jAEGAAmsiBSQAAkAgAiADTA0AIARBgMAEcQ0AIAUgAUH/AXEgAiADayIDQYACIANBgAJJIgEbENQBGiABRQRAA0AgACAFQYACEOgBIANBgAJrIgNB/wFLDQALCyAAIAUgAxDoAQsgBUGAAmokAAsRACAAIAEgAkGGAUGHARDmAQu1GAMSfwF8An4jAEGwBGsiDCQAIAxBADYCLAJAIAG9IhlCAFMEQEEBIRBB5wshEyABmiIBvSEZDAELIARBgBBxBEBBASEQQeoLIRMMAQtB7QtB6AsgBEEBcSIQGyETIBBFIRULAkAgGUKAgICAgICA+P8Ag0KAgICAgICA+P8AUQRAIABBICACIBBBA2oiAyAEQf//e3EQ7AEgACATIBAQ6AEgAEGuHEGDLiAFQSBxIgUbQckjQbIuIAUbIAEgAWIbQQMQ6AEgAEEgIAIgAyAEQYDAAHMQ7AEgAyACIAIgA0gbIQkMAQsgDEEQaiERAkACfwJAIAEgDEEsahDlASIBIAGgIgFEAAAAAAAAAABiBEAgDCAMKAIsIgZBAWs2AiwgBUEgciIOQeEARw0BDAMLIAVBIHIiDkHhAEYNAiAMKAIsIQpBBiADIANBAEgbDAELIAwgBkEdayIKNgIsIAFEAAAAAAAAsEGiIQFBBiADIANBAEgbCyELIAxBMGpBoAJBACAKQQBOG2oiDSEHA0AgBwJ/IAFEAAAAAAAA8EFjIAFEAAAAAAAAAABmcQRAIAGrDAELQQALIgM2AgAgB0EEaiEHIAEgA7ihRAAAAABlzc1BoiIBRAAAAAAAAAAAYg0ACwJAIApBAEwEQCAKIQMgByEGIA0hCAwBCyANIQggCiEDA0BBHSADIANBHU4bIQMCQCAHQQRrIgYgCEkNACADrSEaQgAhGQNAIAYgGUL/////D4MgBjUCACAahnwiGSAZQoCU69wDgCIZQoCU69wDfn0+AgAgBkEEayIGIAhPDQALIBmnIgZFDQAgCEEEayIIIAY2AgALA0AgCCAHIgZJBEAgBkEEayIHKAIARQ0BCwsgDCAMKAIsIANrIgM2AiwgBiEHIANBAEoNAAsLIANBAEgEQCALQRlqQQluQQFqIQ8gDkHmAEYhEgNAQQlBACADayIDIANBCU4bIQkCQCAGIAhNBEAgCCgCACEHDAELQYCU69wDIAl2IRRBfyAJdEF/cyEWQQAhAyAIIQcDQCAHIAMgBygCACIXIAl2ajYCACAWIBdxIBRsIQMgB0EEaiIHIAZJDQALIAgoAgAhByADRQ0AIAYgAzYCACAGQQRqIQYLIAwgDCgCLCAJaiIDNgIsIA0gCCAHRUECdGoiCCASGyIHIA9BAnRqIAYgBiAHa0ECdSAPShshBiADQQBIDQALC0EAIQMCQCAGIAhNDQAgDSAIa0ECdUEJbCEDQQohByAIKAIAIglBCkkNAANAIANBAWohAyAJIAdBCmwiB08NAAsLIAsgA0EAIA5B5gBHG2sgDkHnAEYgC0EAR3FrIgcgBiANa0ECdUEJbEEJa0gEQEEEQaQCIApBAEgbIAxqIAdBgMgAaiIJQQltIg9BAnRqQdAfayEKQQohByAJIA9BCWxrIglBB0wEQANAIAdBCmwhByAJQQFqIglBCEcNAAsLAkAgCigCACISIBIgB24iDyAHbGsiCUUgCkEEaiIUIAZGcQ0AAkAgD0EBcUUEQEQAAAAAAABAQyEBIAdBgJTr3ANHDQEgCCAKTw0BIApBBGstAABBAXFFDQELRAEAAAAAAEBDIQELRAAAAAAAAOA/RAAAAAAAAPA/RAAAAAAAAPg/IAYgFEYbRAAAAAAAAPg/IAkgB0EBdiIURhsgCSAUSRshGAJAIBUNACATLQAAQS1HDQAgGJohGCABmiEBCyAKIBIgCWsiCTYCACABIBigIAFhDQAgCiAHIAlqIgM2AgAgA0GAlOvcA08EQANAIApBADYCACAIIApBBGsiCksEQCAIQQRrIghBADYCAAsgCiAKKAIAQQFqIgM2AgAgA0H/k+vcA0sNAAsLIA0gCGtBAnVBCWwhA0EKIQcgCCgCACIJQQpJDQADQCADQQFqIQMgCSAHQQpsIgdPDQALCyAKQQRqIgcgBiAGIAdLGyEGCwNAIAYiByAITSIJRQRAIAdBBGsiBigCAEUNAQsLAkAgDkHnAEcEQCAEQQhxIQoMAQsgA0F/c0F/IAtBASALGyIGIANKIANBe0pxIgobIAZqIQtBf0F+IAobIAVqIQUgBEEIcSIKDQBBdyEGAkAgCQ0AIAdBBGsoAgAiDkUNAEEKIQlBACEGIA5BCnANAANAIAYiCkEBaiEGIA4gCUEKbCIJcEUNAAsgCkF/cyEGCyAHIA1rQQJ1QQlsIQkgBUFfcUHGAEYEQEEAIQogCyAGIAlqQQlrIgZBACAGQQBKGyIGIAYgC0obIQsMAQtBACEKIAsgAyAJaiAGakEJayIGQQAgBkEAShsiBiAGIAtKGyELC0F/IQkgC0H9////B0H+////ByAKIAtyIhIbSg0BIAsgEkEAR2pBAWohDgJAIAVBX3EiFUHGAEYEQCADIA5B/////wdzSg0DIANBACADQQBKGyEGDAELIBEgAyADQR91IgZzIAZrrSAREOsBIgZrQQFMBEADQCAGQQFrIgZBMDoAACARIAZrQQJIDQALCyAGQQJrIg8gBToAACAGQQFrQS1BKyADQQBIGzoAACARIA9rIgYgDkH/////B3NKDQILIAYgDmoiAyAQQf////8Hc0oNASAAQSAgAiADIBBqIgUgBBDsASAAIBMgEBDoASAAQTAgAiAFIARBgIAEcxDsAQJAAkACQCAVQcYARgRAIAxBEGoiBkEIciEDIAZBCXIhCiANIAggCCANSxsiCSEIA0AgCDUCACAKEOsBIQYCQCAIIAlHBEAgBiAMQRBqTQ0BA0AgBkEBayIGQTA6AAAgBiAMQRBqSw0ACwwBCyAGIApHDQAgDEEwOgAYIAMhBgsgACAGIAogBmsQ6AEgCEEEaiIIIA1NDQALIBIEQCAAQdM8QQEQ6AELIAcgCE0NASALQQBMDQEDQCAINQIAIAoQ6wEiBiAMQRBqSwRAA0AgBkEBayIGQTA6AAAgBiAMQRBqSw0ACwsgACAGQQkgCyALQQlOGxDoASALQQlrIQYgCEEEaiIIIAdPDQMgC0EJSiEDIAYhCyADDQALDAILAkAgC0EASA0AIAcgCEEEaiAHIAhLGyEJIAxBEGoiBkEIciEDIAZBCXIhDSAIIQcDQCANIAc1AgAgDRDrASIGRgRAIAxBMDoAGCADIQYLAkAgByAIRwRAIAYgDEEQak0NAQNAIAZBAWsiBkEwOgAAIAYgDEEQaksNAAsMAQsgACAGQQEQ6AEgBkEBaiEGIAogC3JFDQAgAEHTPEEBEOgBCyAAIAYgCyANIAZrIgYgBiALShsQ6AEgCyAGayELIAdBBGoiByAJTw0BIAtBAE4NAAsLIABBMCALQRJqQRJBABDsASAAIA8gESAPaxDoAQwCCyALIQYLIABBMCAGQQlqQQlBABDsAQsgAEEgIAIgBSAEQYDAAHMQ7AEgBSACIAIgBUgbIQkMAQsgEyAFQRp0QR91QQlxaiEIAkAgA0ELSw0AQQwgA2shBkQAAAAAAAAwQCEYA0AgGEQAAAAAAAAwQKIhGCAGQQFrIgYNAAsgCC0AAEEtRgRAIBggAZogGKGgmiEBDAELIAEgGKAgGKEhAQsgESAMKAIsIgYgBkEfdSIGcyAGa60gERDrASIGRgRAIAxBMDoADyAMQQ9qIQYLIBBBAnIhCyAFQSBxIQ0gDCgCLCEHIAZBAmsiCiAFQQ9qOgAAIAZBAWtBLUErIAdBAEgbOgAAIARBCHEhBiAMQRBqIQcDQCAHIgUCfyABmUQAAAAAAADgQWMEQCABqgwBC0GAgICAeAsiB0GQjwFqLQAAIA1yOgAAIAEgB7ehRAAAAAAAADBAoiEBAkAgBUEBaiIHIAxBEGprQQFHDQACQCAGDQAgA0EASg0AIAFEAAAAAAAAAABhDQELIAVBLjoAASAFQQJqIQcLIAFEAAAAAAAAAABiDQALQX8hCUH9////ByALIBEgCmsiBmoiDWsgA0gNACAAQSAgAiANIANBAmogByAMQRBqIgdrIgUgBUECayADSBsgBSADGyIJaiIDIAQQ7AEgACAIIAsQ6AEgAEEwIAIgAyAEQYCABHMQ7AEgACAHIAUQ6AEgAEEwIAkgBWtBAEEAEOwBIAAgCiAGEOgBIABBICACIAMgBEGAwABzEOwBIAMgAiACIANIGyEJCyAMQbAEaiQAIAkLKQAgASABKAIAQQdqQXhxIgFBEGo2AgAgACABKQMAIAEpAwgQ+QE5AwALiQIAAkAgAAR/IAFB/wBNDQECQEHU7QIoAgAoAgBFBEAgAUGAf3FBgL8DRg0DDAELIAFB/w9NBEAgACABQT9xQYABcjoAASAAIAFBBnZBwAFyOgAAQQIPCyABQYBAcUGAwANHIAFBgLADT3FFBEAgACABQT9xQYABcjoAAiAAIAFBDHZB4AFyOgAAIAAgAUEGdkE/cUGAAXI6AAFBAw8LIAFBgIAEa0H//z9NBEAgACABQT9xQYABcjoAAyAAIAFBEnZB8AFyOgAAIAAgAUEGdkE/cUGAAXI6AAIgACABQQx2QT9xQYABcjoAAUEEDwsLQbjsAkEZNgIAQX8FQQELDwsgACABOgAAQQELEgAgAEUEQEEADwsgACABEPABC50pAQt/IwBBEGsiCyQAAkACQAJAAkACQAJAAkACQAJAIABB9AFNBEBB+O0CKAIAIgZBECAAQQtqQXhxIABBC0kbIgVBA3YiAHYiAUEDcQRAAkAgAUF/c0EBcSAAaiICQQN0IgFBoO4CaiIAIAFBqO4CaigCACIBKAIIIgRGBEBB+O0CIAZBfiACd3E2AgAMAQsgBCAANgIMIAAgBDYCCAsgAUEIaiEAIAEgAkEDdCICQQNyNgIEIAEgAmoiASABKAIEQQFyNgIEDAoLIAVBgO4CKAIAIgdNDQEgAQRAAkBBAiAAdCICQQAgAmtyIAEgAHRxIgBBACAAa3FoIgFBA3QiAEGg7gJqIgIgAEGo7gJqKAIAIgAoAggiBEYEQEH47QIgBkF+IAF3cSIGNgIADAELIAQgAjYCDCACIAQ2AggLIAAgBUEDcjYCBCAAIAVqIgggAUEDdCIBIAVrIgRBAXI2AgQgACABaiAENgIAIAcEQCAHQXhxQaDuAmohAUGM7gIoAgAhAgJ/IAZBASAHQQN2dCIDcUUEQEH47QIgAyAGcjYCACABDAELIAEoAggLIQMgASACNgIIIAMgAjYCDCACIAE2AgwgAiADNgIICyAAQQhqIQBBjO4CIAg2AgBBgO4CIAQ2AgAMCgtB/O0CKAIAIgpFDQEgCkEAIAprcWhBAnRBqPACaigCACICKAIEQXhxIAVrIQMgAiEBA0ACQCABKAIQIgBFBEAgASgCFCIARQ0BCyAAKAIEQXhxIAVrIgEgAyABIANJIgEbIQMgACACIAEbIQIgACEBDAELCyACKAIYIQkgAiACKAIMIgRHBEBBiO4CKAIAGiACKAIIIgAgBDYCDCAEIAA2AggMCQsgAkEUaiIBKAIAIgBFBEAgAigCECIARQ0DIAJBEGohAQsDQCABIQggACIEQRRqIgEoAgAiAA0AIARBEGohASAEKAIQIgANAAsgCEEANgIADAgLQX8hBSAAQb9/Sw0AIABBC2oiAEF4cSEFQfztAigCACIIRQ0AQQAgBWshAwJAAkACQAJ/QQAgBUGAAkkNABpBHyAFQf///wdLDQAaIAVBJiAAQQh2ZyIAa3ZBAXEgAEEBdGtBPmoLIgdBAnRBqPACaigCACIBRQRAQQAhAAwBC0EAIQAgBUEZIAdBAXZrQQAgB0EfRxt0IQIDQAJAIAEoAgRBeHEgBWsiBiADTw0AIAEhBCAGIgMNAEEAIQMgASEADAMLIAAgASgCFCIGIAYgASACQR12QQRxaigCECIBRhsgACAGGyEAIAJBAXQhAiABDQALCyAAIARyRQRAQQAhBEECIAd0IgBBACAAa3IgCHEiAEUNAyAAQQAgAGtxaEECdEGo8AJqKAIAIQALIABFDQELA0AgACgCBEF4cSAFayICIANJIQEgAiADIAEbIQMgACAEIAEbIQQgACgCECIBBH8gAQUgACgCFAsiAA0ACwsgBEUNACADQYDuAigCACAFa08NACAEKAIYIQcgBCAEKAIMIgJHBEBBiO4CKAIAGiAEKAIIIgAgAjYCDCACIAA2AggMBwsgBEEUaiIBKAIAIgBFBEAgBCgCECIARQ0DIARBEGohAQsDQCABIQYgACICQRRqIgEoAgAiAA0AIAJBEGohASACKAIQIgANAAsgBkEANgIADAYLIAVBgO4CKAIAIgRNBEBBjO4CKAIAIQACQCAEIAVrIgFBEE8EQCAAIAVqIgIgAUEBcjYCBCAAIARqIAE2AgAgACAFQQNyNgIEDAELIAAgBEEDcjYCBCAAIARqIgEgASgCBEEBcjYCBEEAIQJBACEBC0GA7gIgATYCAEGM7gIgAjYCACAAQQhqIQAMCAsgBUGE7gIoAgAiAkkEQEGE7gIgAiAFayIBNgIAQZDuAkGQ7gIoAgAiACAFaiICNgIAIAIgAUEBcjYCBCAAIAVBA3I2AgQgAEEIaiEADAgLQQAhACAFQS9qIgMCf0HQ8QIoAgAEQEHY8QIoAgAMAQtB3PECQn83AgBB1PECQoCggICAgAQ3AgBB0PECIAtBDGpBcHFB2KrVqgVzNgIAQeTxAkEANgIAQbTxAkEANgIAQYAgCyIBaiIGQQAgAWsiCHEiASAFTQ0HQbDxAigCACIEBEBBqPECKAIAIgcgAWoiCSAHTQ0IIAQgCUkNCAsCQEG08QItAABBBHFFBEACQAJAAkACQEGQ7gIoAgAiBARAQbjxAiEAA0AgBCAAKAIAIgdPBEAgByAAKAIEaiAESw0DCyAAKAIIIgANAAsLQQAQ9gEiAkF/Rg0DIAEhBkHU8QIoAgAiAEEBayIEIAJxBEAgASACayACIARqQQAgAGtxaiEGCyAFIAZPDQNBsPECKAIAIgAEQEGo8QIoAgAiBCAGaiIIIARNDQQgACAISQ0ECyAGEPYBIgAgAkcNAQwFCyAGIAJrIAhxIgYQ9gEiAiAAKAIAIAAoAgRqRg0BIAIhAAsgAEF/Rg0BIAYgBUEwak8EQCAAIQIMBAtB2PECKAIAIgIgAyAGa2pBACACa3EiAhD2AUF/Rg0BIAIgBmohBiAAIQIMAwsgAkF/Rw0CC0G08QJBtPECKAIAQQRyNgIACyABEPYBIQJBABD2ASEAIAJBf0YNBSAAQX9GDQUgACACTQ0FIAAgAmsiBiAFQShqTQ0FC0Go8QJBqPECKAIAIAZqIgA2AgBBrPECKAIAIABJBEBBrPECIAA2AgALAkBBkO4CKAIAIgMEQEG48QIhAANAIAIgACgCACIBIAAoAgQiBGpGDQIgACgCCCIADQALDAQLQYjuAigCACIAQQAgACACTRtFBEBBiO4CIAI2AgALQQAhAEG88QIgBjYCAEG48QIgAjYCAEGY7gJBfzYCAEGc7gJB0PECKAIANgIAQcTxAkEANgIAA0AgAEEDdCIBQajuAmogAUGg7gJqIgQ2AgAgAUGs7gJqIAQ2AgAgAEEBaiIAQSBHDQALQYTuAiAGQShrIgBBeCACa0EHcUEAIAJBCGpBB3EbIgFrIgQ2AgBBkO4CIAEgAmoiATYCACABIARBAXI2AgQgACACakEoNgIEQZTuAkHg8QIoAgA2AgAMBAsgAC0ADEEIcQ0CIAEgA0sNAiACIANNDQIgACAEIAZqNgIEQZDuAiADQXggA2tBB3FBACADQQhqQQdxGyIAaiIBNgIAQYTuAkGE7gIoAgAgBmoiAiAAayIANgIAIAEgAEEBcjYCBCACIANqQSg2AgRBlO4CQeDxAigCADYCAAwDC0EAIQQMBQtBACECDAMLQYjuAigCACACSwRAQYjuAiACNgIACyACIAZqIQFBuPECIQACQAJAAkACQAJAAkADQCABIAAoAgBHBEAgACgCCCIADQEMAgsLIAAtAAxBCHFFDQELQbjxAiEAA0AgAyAAKAIAIgFPBEAgASAAKAIEaiIEIANLDQMLIAAoAgghAAwACwALIAAgAjYCACAAIAAoAgQgBmo2AgQgAkF4IAJrQQdxQQAgAkEIakEHcRtqIgcgBUEDcjYCBCABQXggAWtBB3FBACABQQhqQQdxG2oiBiAFIAdqIgVrIQAgAyAGRgRAQZDuAiAFNgIAQYTuAkGE7gIoAgAgAGoiADYCACAFIABBAXI2AgQMAwtBjO4CKAIAIAZGBEBBjO4CIAU2AgBBgO4CQYDuAigCACAAaiIANgIAIAUgAEEBcjYCBCAAIAVqIAA2AgAMAwsgBigCBCIDQQNxQQFGBEAgA0F4cSEJAkAgA0H/AU0EQCAGKAIMIgEgBigCCCICRgRAQfjtAkH47QIoAgBBfiADQQN2d3E2AgAMAgsgAiABNgIMIAEgAjYCCAwBCyAGKAIYIQgCQCAGIAYoAgwiAkcEQCAGKAIIIgEgAjYCDCACIAE2AggMAQsCQCAGQRRqIgMoAgAiAQ0AIAZBEGoiAygCACIBDQBBACECDAELA0AgAyEEIAEiAkEUaiIDKAIAIgENACACQRBqIQMgAigCECIBDQALIARBADYCAAsgCEUNAAJAIAYoAhwiAUECdEGo8AJqIgQoAgAgBkYEQCAEIAI2AgAgAg0BQfztAkH87QIoAgBBfiABd3E2AgAMAgsgCEEQQRQgCCgCECAGRhtqIAI2AgAgAkUNAQsgAiAINgIYIAYoAhAiAQRAIAIgATYCECABIAI2AhgLIAYoAhQiAUUNACACIAE2AhQgASACNgIYCyAGIAlqIgYoAgQhAyAAIAlqIQALIAYgA0F+cTYCBCAFIABBAXI2AgQgACAFaiAANgIAIABB/wFNBEAgAEF4cUGg7gJqIQECf0H47QIoAgAiAkEBIABBA3Z0IgBxRQRAQfjtAiAAIAJyNgIAIAEMAQsgASgCCAshACABIAU2AgggACAFNgIMIAUgATYCDCAFIAA2AggMAwtBHyEDIABB////B00EQCAAQSYgAEEIdmciAWt2QQFxIAFBAXRrQT5qIQMLIAUgAzYCHCAFQgA3AhAgA0ECdEGo8AJqIQECQEH87QIoAgAiAkEBIAN0IgRxRQRAQfztAiACIARyNgIAIAEgBTYCAAwBCyAAQRkgA0EBdmtBACADQR9HG3QhAyABKAIAIQIDQCACIgEoAgRBeHEgAEYNAyADQR12IQIgA0EBdCEDIAEgAkEEcWoiBCgCECICDQALIAQgBTYCEAsgBSABNgIYIAUgBTYCDCAFIAU2AggMAgtBhO4CIAZBKGsiAEF4IAJrQQdxQQAgAkEIakEHcRsiAWsiCDYCAEGQ7gIgASACaiIBNgIAIAEgCEEBcjYCBCAAIAJqQSg2AgRBlO4CQeDxAigCADYCACADIARBJyAEa0EHcUEAIARBJ2tBB3EbakEvayIAIAAgA0EQakkbIgFBGzYCBCABQcDxAikCADcCECABQbjxAikCADcCCEHA8QIgAUEIajYCAEG88QIgBjYCAEG48QIgAjYCAEHE8QJBADYCACABQRhqIQADQCAAQQc2AgQgAEEIaiECIABBBGohACACIARJDQALIAEgA0YNAyABIAEoAgRBfnE2AgQgAyABIANrIgJBAXI2AgQgASACNgIAIAJB/wFNBEAgAkF4cUGg7gJqIQACf0H47QIoAgAiAUEBIAJBA3Z0IgJxRQRAQfjtAiABIAJyNgIAIAAMAQsgACgCCAshASAAIAM2AgggASADNgIMIAMgADYCDCADIAE2AggMBAtBHyEAIAJB////B00EQCACQSYgAkEIdmciAGt2QQFxIABBAXRrQT5qIQALIAMgADYCHCADQgA3AhAgAEECdEGo8AJqIQECQEH87QIoAgAiBEEBIAB0IgZxRQRAQfztAiAEIAZyNgIAIAEgAzYCAAwBCyACQRkgAEEBdmtBACAAQR9HG3QhACABKAIAIQQDQCAEIgEoAgRBeHEgAkYNBCAAQR12IQQgAEEBdCEAIAEgBEEEcWoiBigCECIEDQALIAYgAzYCEAsgAyABNgIYIAMgAzYCDCADIAM2AggMAwsgASgCCCIAIAU2AgwgASAFNgIIIAVBADYCGCAFIAE2AgwgBSAANgIICyAHQQhqIQAMBQsgASgCCCIAIAM2AgwgASADNgIIIANBADYCGCADIAE2AgwgAyAANgIIC0GE7gIoAgAiACAFTQ0AQYTuAiAAIAVrIgE2AgBBkO4CQZDuAigCACIAIAVqIgI2AgAgAiABQQFyNgIEIAAgBUEDcjYCBCAAQQhqIQAMAwtBuOwCQTA2AgBBACEADAILAkAgB0UNAAJAIAQoAhwiAEECdEGo8AJqIgEoAgAgBEYEQCABIAI2AgAgAg0BQfztAiAIQX4gAHdxIgg2AgAMAgsgB0EQQRQgBygCECAERhtqIAI2AgAgAkUNAQsgAiAHNgIYIAQoAhAiAARAIAIgADYCECAAIAI2AhgLIAQoAhQiAEUNACACIAA2AhQgACACNgIYCwJAIANBD00EQCAEIAMgBWoiAEEDcjYCBCAAIARqIgAgACgCBEEBcjYCBAwBCyAEIAVBA3I2AgQgBCAFaiICIANBAXI2AgQgAiADaiADNgIAIANB/wFNBEAgA0F4cUGg7gJqIQACf0H47QIoAgAiAUEBIANBA3Z0IgNxRQRAQfjtAiABIANyNgIAIAAMAQsgACgCCAshASAAIAI2AgggASACNgIMIAIgADYCDCACIAE2AggMAQtBHyEAIANB////B00EQCADQSYgA0EIdmciAGt2QQFxIABBAXRrQT5qIQALIAIgADYCHCACQgA3AhAgAEECdEGo8AJqIQECQAJAIAhBASAAdCIGcUUEQEH87QIgBiAIcjYCACABIAI2AgAMAQsgA0EZIABBAXZrQQAgAEEfRxt0IQAgASgCACEFA0AgBSIBKAIEQXhxIANGDQIgAEEddiEGIABBAXQhACABIAZBBHFqIgYoAhAiBQ0ACyAGIAI2AhALIAIgATYCGCACIAI2AgwgAiACNgIIDAELIAEoAggiACACNgIMIAEgAjYCCCACQQA2AhggAiABNgIMIAIgADYCCAsgBEEIaiEADAELAkAgCUUNAAJAIAIoAhwiAEECdEGo8AJqIgEoAgAgAkYEQCABIAQ2AgAgBA0BQfztAiAKQX4gAHdxNgIADAILIAlBEEEUIAkoAhAgAkYbaiAENgIAIARFDQELIAQgCTYCGCACKAIQIgAEQCAEIAA2AhAgACAENgIYCyACKAIUIgBFDQAgBCAANgIUIAAgBDYCGAsCQCADQQ9NBEAgAiADIAVqIgBBA3I2AgQgACACaiIAIAAoAgRBAXI2AgQMAQsgAiAFQQNyNgIEIAIgBWoiBCADQQFyNgIEIAMgBGogAzYCACAHBEAgB0F4cUGg7gJqIQBBjO4CKAIAIQECf0EBIAdBA3Z0IgUgBnFFBEBB+O0CIAUgBnI2AgAgAAwBCyAAKAIICyEGIAAgATYCCCAGIAE2AgwgASAANgIMIAEgBjYCCAtBjO4CIAQ2AgBBgO4CIAM2AgALIAJBCGohAAsgC0EQaiQAIAAL7gsBB38CQCAARQ0AIABBCGsiAiAAQQRrKAIAIgFBeHEiAGohBQJAIAFBAXENACABQQNxRQ0BIAIgAigCACIBayICQYjuAigCAEkNASAAIAFqIQBBjO4CKAIAIAJHBEAgAUH/AU0EQCABQQN2IQEgAigCDCIDIAIoAggiBEYEQEH47QJB+O0CKAIAQX4gAXdxNgIADAMLIAQgAzYCDCADIAQ2AggMAgsgAigCGCEGAkAgAiACKAIMIgFHBEAgAigCCCIDIAE2AgwgASADNgIIDAELAkAgAkEUaiIEKAIAIgMNACACQRBqIgQoAgAiAw0AQQAhAQwBCwNAIAQhByADIgFBFGoiBCgCACIDDQAgAUEQaiEEIAEoAhAiAw0ACyAHQQA2AgALIAZFDQECQCACKAIcIgRBAnRBqPACaiIDKAIAIAJGBEAgAyABNgIAIAENAUH87QJB/O0CKAIAQX4gBHdxNgIADAMLIAZBEEEUIAYoAhAgAkYbaiABNgIAIAFFDQILIAEgBjYCGCACKAIQIgMEQCABIAM2AhAgAyABNgIYCyACKAIUIgNFDQEgASADNgIUIAMgATYCGAwBCyAFKAIEIgFBA3FBA0cNAEGA7gIgADYCACAFIAFBfnE2AgQgAiAAQQFyNgIEIAAgAmogADYCAA8LIAIgBU8NACAFKAIEIgFBAXFFDQACQCABQQJxRQRAQZDuAigCACAFRgRAQZDuAiACNgIAQYTuAkGE7gIoAgAgAGoiADYCACACIABBAXI2AgQgAkGM7gIoAgBHDQNBgO4CQQA2AgBBjO4CQQA2AgAPC0GM7gIoAgAgBUYEQEGM7gIgAjYCAEGA7gJBgO4CKAIAIABqIgA2AgAgAiAAQQFyNgIEIAAgAmogADYCAA8LIAFBeHEgAGohAAJAIAFB/wFNBEAgAUEDdiEBIAUoAgwiAyAFKAIIIgRGBEBB+O0CQfjtAigCAEF+IAF3cTYCAAwCCyAEIAM2AgwgAyAENgIIDAELIAUoAhghBgJAIAUgBSgCDCIBRwRAQYjuAigCABogBSgCCCIDIAE2AgwgASADNgIIDAELAkAgBUEUaiIEKAIAIgMNACAFQRBqIgQoAgAiAw0AQQAhAQwBCwNAIAQhByADIgFBFGoiBCgCACIDDQAgAUEQaiEEIAEoAhAiAw0ACyAHQQA2AgALIAZFDQACQCAFKAIcIgRBAnRBqPACaiIDKAIAIAVGBEAgAyABNgIAIAENAUH87QJB/O0CKAIAQX4gBHdxNgIADAILIAZBEEEUIAYoAhAgBUYbaiABNgIAIAFFDQELIAEgBjYCGCAFKAIQIgMEQCABIAM2AhAgAyABNgIYCyAFKAIUIgNFDQAgASADNgIUIAMgATYCGAsgAiAAQQFyNgIEIAAgAmogADYCACACQYzuAigCAEcNAUGA7gIgADYCAA8LIAUgAUF+cTYCBCACIABBAXI2AgQgACACaiAANgIACyAAQf8BTQRAIABBeHFBoO4CaiEBAn9B+O0CKAIAIgNBASAAQQN2dCIAcUUEQEH47QIgACADcjYCACABDAELIAEoAggLIQAgASACNgIIIAAgAjYCDCACIAE2AgwgAiAANgIIDwtBHyEEIABB////B00EQCAAQSYgAEEIdmciAWt2QQFxIAFBAXRrQT5qIQQLIAIgBDYCHCACQgA3AhAgBEECdEGo8AJqIQcCQAJAAkBB/O0CKAIAIgNBASAEdCIBcUUEQEH87QIgASADcjYCACAHIAI2AgAgAiAHNgIYDAELIABBGSAEQQF2a0EAIARBH0cbdCEEIAcoAgAhAQNAIAEiAygCBEF4cSAARg0CIARBHXYhASAEQQF0IQQgAyABQQRxaiIHQRBqKAIAIgENAAsgByACNgIQIAIgAzYCGAsgAiACNgIMIAIgAjYCCAwBCyADKAIIIgAgAjYCDCADIAI2AgggAkEANgIYIAIgAzYCDCACIAA2AggLQZjuAkGY7gIoAgBBAWsiAEF/IAAbNgIACwuJCAELfyAARQRAIAEQ8gEPCyABQUBPBEBBuOwCQTA2AgBBAA8LAn9BECABQQtqQXhxIAFBC0kbIQUgAEEIayIEKAIEIglBeHEhAwJAIAlBA3FFBEBBACAFQYACSQ0CGiAFQQRqIANNBEAgBCECIAMgBWtB2PECKAIAQQF0TQ0CC0EADAILIAMgBGohBgJAIAMgBU8EQCADIAVrIgNBEEkNASAEIAlBAXEgBXJBAnI2AgQgBCAFaiICIANBA3I2AgQgBiAGKAIEQQFyNgIEIAIgAxD1AQwBC0GQ7gIoAgAgBkYEQEGE7gIoAgAgA2oiCCAFTQ0CIAQgCUEBcSAFckECcjYCBCAEIAVqIgMgCCAFayICQQFyNgIEQYTuAiACNgIAQZDuAiADNgIADAELQYzuAigCACAGRgRAQYDuAigCACADaiIDIAVJDQICQCADIAVrIgJBEE8EQCAEIAlBAXEgBXJBAnI2AgQgBCAFaiIIIAJBAXI2AgQgAyAEaiIDIAI2AgAgAyADKAIEQX5xNgIEDAELIAQgCUEBcSADckECcjYCBCADIARqIgIgAigCBEEBcjYCBEEAIQILQYzuAiAINgIAQYDuAiACNgIADAELIAYoAgQiCEECcQ0BIAhBeHEgA2oiCiAFSQ0BIAogBWshDAJAIAhB/wFNBEAgBigCDCIDIAYoAggiAkYEQEH47QJB+O0CKAIAQX4gCEEDdndxNgIADAILIAIgAzYCDCADIAI2AggMAQsgBigCGCELAkAgBiAGKAIMIgdHBEBBiO4CKAIAGiAGKAIIIgIgBzYCDCAHIAI2AggMAQsCQCAGQRRqIggoAgAiAg0AIAZBEGoiCCgCACICDQBBACEHDAELA0AgCCEDIAIiB0EUaiIIKAIAIgINACAHQRBqIQggBygCECICDQALIANBADYCAAsgC0UNAAJAIAYoAhwiA0ECdEGo8AJqIgIoAgAgBkYEQCACIAc2AgAgBw0BQfztAkH87QIoAgBBfiADd3E2AgAMAgsgC0EQQRQgCygCECAGRhtqIAc2AgAgB0UNAQsgByALNgIYIAYoAhAiAgRAIAcgAjYCECACIAc2AhgLIAYoAhQiAkUNACAHIAI2AhQgAiAHNgIYCyAMQQ9NBEAgBCAJQQFxIApyQQJyNgIEIAQgCmoiAiACKAIEQQFyNgIEDAELIAQgCUEBcSAFckECcjYCBCAEIAVqIgMgDEEDcjYCBCAEIApqIgIgAigCBEEBcjYCBCADIAwQ9QELIAQhAgsgAgsiAgRAIAJBCGoPCyABEPIBIgRFBEBBAA8LIAQgAEF8QXggAEEEaygCACICQQNxGyACQXhxaiICIAEgASACSxsQ0gEaIAAQ8wEgBAuqCwEGfyAAIAFqIQUCQAJAIAAoAgQiAkEBcQ0AIAJBA3FFDQEgACgCACICIAFqIQECQCAAIAJrIgBBjO4CKAIARwRAIAJB/wFNBEAgAkEDdiECIAAoAggiBCAAKAIMIgNHDQJB+O0CQfjtAigCAEF+IAJ3cTYCAAwDCyAAKAIYIQYCQCAAIAAoAgwiAkcEQEGI7gIoAgAaIAAoAggiAyACNgIMIAIgAzYCCAwBCwJAIABBFGoiBCgCACIDDQAgAEEQaiIEKAIAIgMNAEEAIQIMAQsDQCAEIQcgAyICQRRqIgQoAgAiAw0AIAJBEGohBCACKAIQIgMNAAsgB0EANgIACyAGRQ0CAkAgACgCHCIEQQJ0QajwAmoiAygCACAARgRAIAMgAjYCACACDQFB/O0CQfztAigCAEF+IAR3cTYCAAwECyAGQRBBFCAGKAIQIABGG2ogAjYCACACRQ0DCyACIAY2AhggACgCECIDBEAgAiADNgIQIAMgAjYCGAsgACgCFCIDRQ0CIAIgAzYCFCADIAI2AhgMAgsgBSgCBCICQQNxQQNHDQFBgO4CIAE2AgAgBSACQX5xNgIEIAAgAUEBcjYCBCAFIAE2AgAPCyAEIAM2AgwgAyAENgIICwJAIAUoAgQiAkECcUUEQEGQ7gIoAgAgBUYEQEGQ7gIgADYCAEGE7gJBhO4CKAIAIAFqIgE2AgAgACABQQFyNgIEIABBjO4CKAIARw0DQYDuAkEANgIAQYzuAkEANgIADwtBjO4CKAIAIAVGBEBBjO4CIAA2AgBBgO4CQYDuAigCACABaiIBNgIAIAAgAUEBcjYCBCAAIAFqIAE2AgAPCyACQXhxIAFqIQECQCACQf8BTQRAIAJBA3YhAiAFKAIMIgMgBSgCCCIERgRAQfjtAkH47QIoAgBBfiACd3E2AgAMAgsgBCADNgIMIAMgBDYCCAwBCyAFKAIYIQYCQCAFIAUoAgwiAkcEQEGI7gIoAgAaIAUoAggiAyACNgIMIAIgAzYCCAwBCwJAIAVBFGoiAygCACIEDQAgBUEQaiIDKAIAIgQNAEEAIQIMAQsDQCADIQcgBCICQRRqIgMoAgAiBA0AIAJBEGohAyACKAIQIgQNAAsgB0EANgIACyAGRQ0AAkAgBSgCHCIEQQJ0QajwAmoiAygCACAFRgRAIAMgAjYCACACDQFB/O0CQfztAigCAEF+IAR3cTYCAAwCCyAGQRBBFCAGKAIQIAVGG2ogAjYCACACRQ0BCyACIAY2AhggBSgCECIDBEAgAiADNgIQIAMgAjYCGAsgBSgCFCIDRQ0AIAIgAzYCFCADIAI2AhgLIAAgAUEBcjYCBCAAIAFqIAE2AgAgAEGM7gIoAgBHDQFBgO4CIAE2AgAPCyAFIAJBfnE2AgQgACABQQFyNgIEIAAgAWogATYCAAsgAUH/AU0EQCABQXhxQaDuAmohAgJ/QfjtAigCACIDQQEgAUEDdnQiAXFFBEBB+O0CIAEgA3I2AgAgAgwBCyACKAIICyEBIAIgADYCCCABIAA2AgwgACACNgIMIAAgATYCCA8LQR8hBCABQf///wdNBEAgAUEmIAFBCHZnIgJrdkEBcSACQQF0a0E+aiEECyAAIAQ2AhwgAEIANwIQIARBAnRBqPACaiEHAkACQEH87QIoAgAiA0EBIAR0IgJxRQRAQfztAiACIANyNgIAIAcgADYCACAAIAc2AhgMAQsgAUEZIARBAXZrQQAgBEEfRxt0IQQgBygCACECA0AgAiIDKAIEQXhxIAFGDQIgBEEddiECIARBAXQhBCADIAJBBHFqIgdBEGooAgAiAg0ACyAHIAA2AhAgACADNgIYCyAAIAA2AgwgACAANgIIDwsgAygCCCIBIAA2AgwgAyAANgIIIABBADYCGCAAIAM2AgwgACABNgIICwtSAQJ/QbTbAigCACIBIABBB2pBeHEiAmohAAJAIAJBACAAIAFNGw0AIAA/AEEQdEsEQCAAECBFDQELQbTbAiAANgIAIAEPC0G47AJBMDYCAEF/C1ABAX4CQCADQcAAcQRAIAEgA0FAaq2GIQJCACEBDAELIANFDQAgAiADrSIEhiABQcAAIANrrYiEIQIgASAEhiEBCyAAIAE3AwAgACACNwMIC1ABAX4CQCADQcAAcQRAIAIgA0FAaq2IIQFCACECDAELIANFDQAgAkHAACADa62GIAEgA60iBIiEIQEgAiAEiCECCyAAIAE3AwAgACACNwMIC9MDAgJ+An8jAEEgayIEJAACQCABQv///////////wCDIgNCgICAgICAwIA8fSADQoCAgICAgMD/wwB9VARAIAFCBIYgAEI8iIQhAyAAQv//////////D4MiAEKBgICAgICAgAhaBEAgA0KBgICAgICAgMAAfCECDAILIANCgICAgICAgIBAfSECIABCgICAgICAgIAIUg0BIAIgA0IBg3whAgwBCyAAUCADQoCAgICAgMD//wBUIANCgICAgICAwP//AFEbRQRAIAFCBIYgAEI8iIRC/////////wODQoCAgICAgID8/wCEIQIMAQtCgICAgICAgPj/ACECIANC////////v//DAFYNAEIAIQIgA0IwiKciBUGR9wBJDQAgBEEQaiAAIAFC////////P4NCgICAgICAwACEIgIgBUGB9wBrEPcBIAQgACACQYH4ACAFaxD4ASAEKQMIQgSGIAQpAwAiAEI8iIQhAiAEKQMQIAQpAxiEQgBSrSAAQv//////////D4OEIgBCgYCAgICAgIAIWgRAIAJCAXwhAgwBCyAAQoCAgICAgICACFINACACQgGDIAJ8IQILIARBIGokACACIAFCgICAgICAgICAf4OEvwuuDAEGfyMAQRBrIgQkACAEIAA2AgwCQCAAQdMBTQRA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BSQAAn8CQAJAQdYqIAEsAAAQ/wFFBEBBuOwCQRw2AgAMAQtBmAkQ8gEiAw0BC0EADAELIANBAEGQARDUARogAUErEP8BRQRAIANBCEEEIAEtAABB8gBGGzYCAAsCQCABLQAAQeEARwRAIAMoAgAhAQwBCyAEQQNBABAiIgFBgAhxRQRAIAUgAUGACHKsNwMQIARBBCAFQRBqECIaCyADIAMoAgBBgAFyIgE2AgALIANBfzYCUCADQYAINgIwIAMgBDYCPCADIANBmAFqNgIsAkAgAUEIcQ0AIAUgBUEYaq03AwAgBEGTqAEgBRAjDQAgA0EKNgJQCyADQYgBNgIoIANBhAE2AiQgA0GJATYCICADQYoBNgIMQb3sAi0AAEUEQCADQX82AkwLIANBpOQCKAIANgI4QaTkAigCACIBBEAgASADNgI0C0Gk5AIgAzYCACADCyEDIAVBIGokACADDQEgBBAlGgtBACEDCyAGQRBqJAAgACADIgE2AkAgA0UNACAAIAI2AlggAkECcUUNASABQgBBAhCEAkUNASAAKAJAEIYCGiAAQQA2AkALQQAPCyAAC4oDAQR/IwBBEGsiAiQAIABBqJIBNgIAIABBBGoQvAQgAEIANwIYIABCADcCECAAQgA3AgggAEEANgIoIABCADcCICAAQciTATYCACAAQTRqQQBBLxDUARogAiAAKAIEIgE2AgwgASABKAIEQQFqNgIEIwAhAQZAIAIoAgxBiPQCELQEELsEIQMZIAEkABDwBQALIAIoAgwiASABKAIEQQFrIgQ2AgQgBEF/RgRAIAEgASgCACgCCBECAAsGQCADBEAgAkEIaiIDIAAoAgQiATYCACABIAEoAgRBAWo2AgQGQCADELsCIQEZIAIkACACKAIIIgEgASgCBEEBayIDNgIEIANBf0YEQCABIAEoAgAoAggRAgALCQALIAAgATYCRCACKAIIIgEgASgCBEEBayIDNgIEIANBf0YEQCABIAEoAgAoAggRAgALIAAgACgCRCIBIAEoAgAoAhwRAQA6AGILIABBAEGAICAAKAIAKAIMEQQAGhkgAiQAIAAQiwIaCQALIAJBEGokACAACwsAIABBiPQCEJQDC3ABAn8jACEBIABByJMBNgIABkAGQCAAEL0CGgcAIQIgASQAIAIQ6QUaEOoFCxkgASQAEPAFAAsCQCAALQBgRQ0AIAAoAiAiAUUNACABEPMBCwJAIAAtAGFFDQAgACgCOCIBRQ0AIAEQ8wELIAAQiwILmQEBBH8jAEEQayICJAAgACgCQCIBBH8gAkGLATYCBCACQQhqIAEgAkEEahC+AiEBBkAgACAAKAIAKAIYEQEAIQQgASgCACEDIAFBADYCACADEIYCIQMgAEEANgJAIABBAEEAIAAoAgAoAgwRBAAaGSACJAAgARDAAgkACyABEMACQQAgACADIARyGwVBAAshACACQRBqJAAgAAs0AQF/IwBBEGsiAyQAIAMgATYCDCAAIAMoAgw2AgAgAEEEaiACKAIANgIAIANBEGokACAACw0AIAAQvAIaIAAQ8wELNQECfyAAKAIAIQEgAEEANgIAIAEEQCMAIQIGQCABIABBBGooAgARAQAaGSACJAAQ8AUACwsLzQYBB38jAEEQayIFJAACQAJAIAAoAkBFBEBBfyEEDAELIAAoAlxBCHEiBEUEQCAAQQA2AhwgAEEANgIUIABBADYCGAJAIAAtAGIEQCAAIAAoAiAiASAAKAI0aiICNgIQDAELIAAgACgCOCIBIAAoAjxqIgI2AhALIAAgAjYCDCAAIAE2AgggAEEINgJcCyAAKAIMRQRAIAAgBUEQaiIBNgIQIAAgATYCDCAAIAVBD2o2AggLIAQEQCAAKAIQIQMgACgCCCEEIAVBBDYCBCAFIAMgBGtBAm02AggjAEEQayIDJAAgBUEEaiIEKAIAIAVBCGoiASgCAEkhAiADQRBqJAAgBCABIAIbKAIAIQMLQX8hBAJAIAAoAgwgACgCEEYEQCAAKAIIIAAoAhAgA2sgAxDTARogAC0AYgRAIAMgACgCCCIBaiAAKAIQIAEgA2prIAAoAkAQiAIiAUUNAiAAIAMgACgCCCIEaiIDIAFqNgIQIAAgAzYCDCAAIAQ2AgggACgCDC0AACEEDAILAn8gACgCKCICIAAoAiQiAUYEQCABDAELIAAoAiAgASACIAFrENMBGiAAKAIkIQEgACgCKAshBiAAIAAoAiAiAiAGIAFraiIBNgIkIAAgAkEIIAAoAjQgAiAAQSxqRhtqIgI2AiggBSAAKAI8IANrNgIIIAUgAiABazYCBCMAQRBrIgEkACAFQQRqIgIoAgAgBUEIaiIGKAIASSEHIAFBEGokACACIAYgBxsoAgAhASAAIAApAkg3AlAgACgCJCABIAAoAkAQiAIiAkUNASAAKAJEIgFFDQMgACAAKAIkIAJqIgI2AigCQCABIABByABqIAAoAiAgAiAAQSRqIAMgACgCCCICaiAAKAI8IAJqIAVBCGogASgCACgCEBEOAEEDRgRAIAAoAiAhAyAAIAAoAig2AhAgACADNgIMIAAgAzYCCAwBCyAFKAIIIAMgACgCCGpGDQIgACgCCCEEIAAgBSgCCDYCECAAIAMgBGo2AgwgACAENgIICyAAKAIMLQAAIQQMAQsgACgCDC0AACEECyAAKAIIIAVBD2pHDQAgAEEANgIQIABBADYCDCAAQQA2AggLIAVBEGokACAEDwsQwgIACygBAX9BBBDkBSIAQdCNAjYCACAAQfSQAjYCACAAQaSRAkGMARDmBQALdwACQCAAKAJARQ0AIAAoAgggACgCDE8NACABQX9GBEAgACAAKAIMQQFrNgIMIAFBACABQX9HGw8LIAAtAFhBEHFFBEAgACgCDEEBay0AACABQf8BcUcNAQsgACAAKAIMQQFrNgIMIAAoAgwgAcA6AAAgAQ8LQX8L5wQBBn8jAEEQayIDJAACfwJAIAAoAkBFDQAgAC0AXEEQcUUEQCAAQQA2AhAgAEEANgIMIABBADYCCAJAIAAoAjQiBUEJTwRAIAAtAGIEQCAAIAAoAiAiAiAFakEBazYCHCAAIAI2AhQgACACNgIYDAILIAAgACgCOCICIAAoAjxqQQFrNgIcIAAgAjYCFCAAIAI2AhgMAQsgAEEANgIcIABBADYCFCAAQQA2AhgLIABBEDYCXAsgACgCFCEFIAAoAhwhBiABQX9HBEAgACgCGEUEQCAAIANBEGo2AhwgACADQQ9qIgI2AhQgACACNgIYCyAAKAIYIAHAOgAAIAAgACgCGEEBajYCGAsgACgCGCAAKAIURwRAAkAgAC0AYgRAIAAoAhQiAkEBIAAoAhggAmsiAiAAKAJAEN4BIAJHDQMMAQsgAyAAKAIgNgIIIABByABqIQcDQCAAKAJEIgIEQCACIAcgACgCFCAAKAIYIANBBGogACgCICIEIAQgACgCNGogA0EIaiACKAIAKAIMEQ4AIQIgACgCFCADKAIERg0EIAJBA0YEQCAAKAIUIgJBASAAKAIYIAJrIgIgACgCQBDeASACRw0FDAMLIAJBAUsNBCAAKAIgIgRBASADKAIIIARrIgQgACgCQBDeASAERw0EIAJBAUcNAiADKAIEIQIgACAAKAIYNgIcIAAgAjYCFCAAIAI2AhggACAAKAIYIAAoAhwgACgCFGtqNgIYDAELCxDCAgALIAAgBjYCHCAAIAU2AhQgACAFNgIYCyABQQAgAUF/RxsMAQtBfwshACADQRBqJAAgAAvqAgEEfyMAQRBrIgQkACAEIAI2AgwgAEEANgIQIABBADYCDCAAQQA2AgggAEEANgIcIABBADYCFCAAQQA2AhgCQCAALQBgRQ0AIAAoAiAiA0UNACADEPMBCwJAIAAtAGFFDQAgACgCOCIDRQ0AIAMQ8wELIAAgAjYCNAJAAkACQCACQQlPBEAgAC0AYiEDAkAgAUUNACADRQ0AIABBADoAYCAAIAE2AiAMAwsgAhCoBSECIABBAToAYCAAIAI2AiAMAQsgAEEAOgBgIABBCDYCNCAAIABBLGo2AiAgAC0AYiEDCyADDQAgBEEINgIIIwBBEGsiAiQAIARBDGoiAygCACAEQQhqIgUoAgBIIQYgAkEQaiQAIAAgBSADIAYbKAIAIgM2AjwgAQRAQQAhAiADQQdLDQILQQEhAiADEKgFIQEMAQtBACEBIABBADYCPEEAIQILIAAgAjoAYSAAIAE2AjggBEEQaiQAIAAL+wEBAX8jAEEQayIEJAAgASgCRCIFBEAgBSAFKAIAKAIYEQEAIQUCQAJAAkAgASgCQEUNACAFQQBMIAJCAFJxDQAgASABKAIAKAIYEQEARQ0BCyAAQn83AwggAEIANwMADAELIANBA08EQCAAQn83AwggAEIANwMADAELIAEoAkAgBa0gAn5CACAFQQBKGyADEIQCBEAgAEJ/NwMIIABCADcDAAwBCyAAAn4gASgCQCIDKAJMQQBIBEAgAxCJAgwBCyADEIkCCzcDCCAAQgA3AwAgBCABKQJIIgI3AwAgBCACNwMIIAAgBCkCADcDAAsgBEEQaiQADwsQwgIAC4oBACMAQRBrIgMkAAJAAkAgASgCQARAIAEgASgCACgCGBEBAEUNAQsgAEJ/NwMIIABCADcDAAwBCyABKAJAIAIpAwhBABCEAgRAIABCfzcDCCAAQgA3AwAMAQsgAyACKQMANwIIIAEgAykDCDcCSCAAIAIpAwg3AwggACACKQMANwMACyADQRBqJAAL9QMCBH8BfiMAQRBrIgMkAAJAIAAoAkBFDQACQCAAKAJEIgQEQCAAKAJcIgJBEHEEQCAAKAIYIAAoAhRHBEBBfyEBIABBfyAAKAIAKAI0EQMAQX9GDQQLIABByABqIQEDQCAAKAJEIgQgASAAKAIgIgIgAiAAKAI0aiADQQxqIAQoAgAoAhQRCQAhBCAAKAIgIgJBASADKAIMIAJrIgIgACgCQBDeASACRw0DAkAgBEEBaw4CAQQACwtBACEBIAAoAkAQhQJFDQMMAgsgAkEIcUUNAiADIAApAlA3AwACfwJAAkAgAC0AYgRAIAAoAhAgACgCDGusIQUMAQsgBCAEKAIAKAIYEQEAIQEgACgCKCAAKAIka6whBSABQQBKBEAgACgCECAAKAIMayABbKwgBXwhBQwBCyAAKAIMIAAoAhBHDQELQQAMAQsgACgCRCIBIAMgACgCICAAKAIkIAAoAgwgACgCCGsgASgCACgCIBEJACEBIAAoAiQgASAAKAIgamusIAV8IQVBAQshASAAKAJAQgAgBX1BARCEAg0BIAEEQCAAIAMpAwA3AkgLIAAgACgCICIBNgIoIAAgATYCJEEAIQEgAEEANgIQIABBADYCDCAAQQA2AgggAEEANgJcDAILEMICAAtBfyEBCyADQRBqJAAgAQupAgEBfyAAIAAoAgAoAhgRAQAaIAAgARC7AiIBNgJEIAAtAGIhAiAAIAEgASgCACgCHBEBACIBOgBiIAEgAkcEQCAAQQA2AhAgAEEANgIMIABBADYCCCAAQQA2AhwgAEEANgIUIABBADYCGCAALQBgIQEgAC0AYgRAAkAgAUUNACAAKAIgIgFFDQAgARDzAQsgACAALQBhOgBgIAAgACgCPDYCNCAAKAI4IQEgAEIANwI4IAAgATYCICAAQQA6AGEPCwJAIAENACAAKAIgIgEgAEEsakYNACAAQQA6AGEgACABNgI4IAAgACgCNCIBNgI8IAEQqAUhASAAQQE6AGAgACABNgIgDwsgACAAKAI0IgE2AjwgARCoBSEBIABBAToAYSAAIAE2AjgLCwoAIAAQsQEQ8wELEwAgACAAKAIAQQxrKAIAahCxAQsTACAAIAAoAgBBDGsoAgBqEMoCCwoAIAAQrAEQ8wELEwAgACAAKAIAQQxrKAIAahCsAQsTACAAIAAoAgBBDGsoAgBqEM0CC4EBAQJ/IwBBEGsiBCQAIwBBIGsiAyQAIANBGGogACABENECIANBEGogAygCGCADKAIcIAIQ0gIgAyAAIAMoAhAgAGtqNgIMIAMgAiADKAIUIAJrajYCCCAEIAMoAgw2AgggBCADKAIINgIMIANBIGokACAEKAIMIQAgBEEQaiQAIAALNgEBfyMAQRBrIgMkACADIAE2AgwgAyACNgIIIAAgAygCDDYCACAAIAMoAgg2AgQgA0EQaiQAC1UBAn8jAEEQayIEJAAgAiABayEFIAEgAkcEQCADIAEgBRDTARoLIAQgASAFajYCDCAEIAMgBWo2AgggACAEKAIMNgIAIAAgBCgCCDYCBCAEQRBqJAALGQAjACEABkAgAUEBENQCGSAAJAAQ8AUACwslACABQQhLBEAjACEBBkAgABDzARkgASQAEPAFAAsPCyAAEPMBCxkAIAEgAhDWAiEBIAAgAjYCBCAAIAE2AgALCQAgAUEBENcCC3QBAX8gAUEISwRAQQQgASABQQRNGyEBQQEgACAAQQFNGyEAAkADQCABIAAQqQUiAg0BQcyCAygCACICBEAgAhEMAAwBCwtBBBDkBSIAQdCNAjYCACAAQaiNAjYCACAAQZyOAkHlABDmBQALIAIPCyAAEKgFC0sBAn8gACgCACIBBEACfyABKAIMIgIgASgCEEYEQCABIAEoAgAoAiQRAQAMAQsgAi0AAAtBf0cEQCAAKAIARQ8LIABBADYCAAtBAQtLAQJ/IAAoAgAiAQRAAn8gASgCDCICIAEoAhBGBEAgASABKAIAKAIkEQEADAELIAIoAgALQX9HBEAgACgCAEUPCyAAQQA2AgALQQELBQBB9hwLGgAgAkEBRwRAIAAgAhDcBQ8LIABB8xYQMhoLBwAgABCOBgsNACAAEI4GGiAAEPMBC9wBAgJ/AX4gACAAKAIYRSABciIBNgIQIAAoAhQgAXEEQCMAQRBrIgEkAEEQEOQFIQIjAEEQayIDJAAgA0EIaiEAQejxAi0AAEUEQEHo8QJBAToAAAsgAEG42wI2AgQgAEEBNgIAIAEgAykDCDcCBCADQRBqJAAgAUEBOgAPBkAjAEEQayIAJAAgACABKQIEIgQ3AwAgACAENwMIIAIgAEH8GBDjBUGkmQE2AgAgAEEQaiQAIAFBADoADyACQfiZAUG2ARDmBRkgASQAIAEtAA8EQCACEOUFCwkACwALC6QBAQN/IwAhAiAAQbiZATYCAAZAAkAgACgCKCEBA0AgAUUNAUEAIAAgAUEBayIBQQJ0IgMgACgCJGooAgAgACgCICADaigCABEFAAwACwALGSACJAAQ8AUACyAAKAIcIgEgASgCBEEBayICNgIEIAJBf0YEQCABIAEoAgAoAggRAgALIAAoAiAQ8wEgACgCJBDzASAAKAIwEPMBIAAoAjwQ8wEgAAsNACAAEN8CGiAAEPMBC0AAIABBADYCFCAAIAE2AhggAEEANgIMIABCgqCAgOAANwIEIAAgAUU2AhAgAEEgakEAQSgQ1AEaIABBHGoQvAQLHgAgACAAKAIQQQFyNgIQIAAtABRBAXEEQBDrBQALC0cBAn8gACABNwNwIAAgACgCLCAAKAIEIgNrrDcDeCAAKAIIIQICQCABUA0AIAIgA2usIAFXDQAgAyABp2ohAgsgACACNgJoC4wCAgN/An4CQCAAKQNwIgRCAFIgBCAAKQN4IAAoAgQiASAAKAIsIgJrrHwiBVdxRQRAIwBBEGsiAiQAQX8hAQJAIAAQhwINACAAIAJBD2pBASAAKAIgEQQAQQFHDQAgAi0ADyEBCyACQRBqJAAgASIDQQBODQEgACgCBCEBIAAoAiwhAgsgAEJ/NwNwIAAgATYCaCAAIAUgAiABa6x8NwN4QX8PCyAFQgF8IQUgACgCBCEBIAAoAgghAgJAIAApA3AiBFANACAEIAV9IgQgAiABa6xZDQAgASAEp2ohAgsgACACNgJoIAAgBSAAKAIsIgAgAWusfDcDeCAAIAFPBEAgAUEBayADOgAACyADC38CAn8BfiMAQRBrIgMkACAAAn4gAUUEQEIADAELIAMgASABQR91IgJzIAJrIgKtQgAgAmciAkHRAGoQ9wEgAykDCEKAgICAgIDAAIVBnoABIAJrrUIwhnwgAUGAgICAeHGtQiCGhCEEIAMpAwALNwMAIAAgBDcDCCADQRBqJAALyQoCBX8PfiMAQeAAayIFJAAgBEL///////8/gyEMIAIgBIVCgICAgICAgICAf4MhCiACQv///////z+DIg1CIIghDiAEQjCIp0H//wFxIQcCQAJAIAJCMIinQf//AXEiCUH//wFrQYKAfk8EQCAHQf//AWtBgYB+Sw0BCyABUCACQv///////////wCDIgtCgICAgICAwP//AFQgC0KAgICAgIDA//8AURtFBEAgAkKAgICAgIAghCEKDAILIANQIARC////////////AIMiAkKAgICAgIDA//8AVCACQoCAgICAgMD//wBRG0UEQCAEQoCAgICAgCCEIQogAyEBDAILIAEgC0KAgICAgIDA//8AhYRQBEAgAiADhFAEQEKAgICAgIDg//8AIQpCACEBDAMLIApCgICAgICAwP//AIQhCkIAIQEMAgsgAyACQoCAgICAgMD//wCFhFAEQCABIAuEIQJCACEBIAJQBEBCgICAgICA4P//ACEKDAMLIApCgICAgICAwP//AIQhCgwCCyABIAuEUARAQgAhAQwCCyACIAOEUARAQgAhAQwCCyALQv///////z9YBEAgBUHQAGogASANIAEgDSANUCIGG3kgBkEGdK18pyIGQQ9rEPcBQRAgBmshBiAFKQNYIg1CIIghDiAFKQNQIQELIAJC////////P1YNACAFQUBrIAMgDCADIAwgDFAiCBt5IAhBBnStfKciCEEPaxD3ASAGIAhrQRBqIQYgBSkDSCEMIAUpA0AhAwsgA0IPhiILQoCA/v8PgyICIAFCIIgiBH4iECALQiCIIhMgAUL/////D4MiAX58Ig9CIIYiESABIAJ+fCILIBFUrSACIA1C/////w+DIg1+IhUgBCATfnwiESAMQg+GIhIgA0IxiIRC/////w+DIgMgAX58IhQgDyAQVK1CIIYgD0IgiIR8Ig8gAiAOQoCABIQiDH4iFiANIBN+fCIOIBJCIIhCgICAgAiEIgIgAX58IhAgAyAEfnwiEkIghnwiF3whASAHIAlqIAZqQf//AGshBgJAIAIgBH4iGCAMIBN+fCIEIBhUrSAEIAQgAyANfnwiBFatfCACIAx+fCAEIAQgESAVVK0gESAUVq18fCIEVq18IAMgDH4iAyACIA1+fCICIANUrUIghiACQiCIhHwgBCACQiCGfCICIARUrXwgAiACIBAgElatIA4gFlStIA4gEFatfHxCIIYgEkIgiIR8IgJWrXwgAiACIA8gFFStIA8gF1atfHwiAlatfCIEQoCAgICAgMAAg0IAUgRAIAZBAWohBgwBCyALQj+IIQMgBEIBhiACQj+IhCEEIAJCAYYgAUI/iIQhAiALQgGGIQsgAyABQgGGhCEBCyAGQf//AU4EQCAKQoCAgICAgMD//wCEIQpCACEBDAELAn4gBkEATARAQQEgBmsiB0H/AE0EQCAFQTBqIAsgASAGQf8AaiIGEPcBIAVBIGogAiAEIAYQ9wEgBUEQaiALIAEgBxD4ASAFIAIgBCAHEPgBIAUpAzAgBSkDOIRCAFKtIAUpAyAgBSkDEISEIQsgBSkDKCAFKQMYhCEBIAUpAwAhAiAFKQMIDAILQgAhAQwCCyAEQv///////z+DIAatQjCGhAsgCoQhCiALUCABQgBZIAFCgICAgICAgICAf1EbRQRAIAogAkIBfCIBUK18IQoMAQsgCyABQoCAgICAgICAgH+FhEIAUgRAIAIhAQwBCyAKIAIgAkIBg3wiASACVK18IQoLIAAgATcDACAAIAo3AwggBUHgAGokAAvMCQIEfwV+IwBB8ABrIgYkACAEQv///////////wCDIQkCQAJAIAFQIgUgAkL///////////8AgyIKQoCAgICAgMD//wB9QoCAgICAgMCAgH9UIApQG0UEQCADQgBSIAlCgICAgICAwP//AH0iC0KAgICAgIDAgIB/ViALQoCAgICAgMCAgH9RGw0BCyAFIApCgICAgICAwP//AFQgCkKAgICAgIDA//8AURtFBEAgAkKAgICAgIAghCEEIAEhAwwCCyADUCAJQoCAgICAgMD//wBUIAlCgICAgICAwP//AFEbRQRAIARCgICAgICAIIQhBAwCCyABIApCgICAgICAwP//AIWEUARAQoCAgICAgOD//wAgAiABIAOFIAIgBIVCgICAgICAgICAf4WEUCIFGyEEQgAgASAFGyEDDAILIAMgCUKAgICAgIDA//8AhYRQDQEgASAKhFAEQCADIAmEQgBSDQIgASADgyEDIAIgBIMhBAwCCyADIAmEQgBSDQAgASEDIAIhBAwBCyADIAEgASADVCAJIApWIAkgClEbIggbIQogBCACIAgbIgtC////////P4MhCSACIAQgCBsiAkIwiKdB//8BcSEHIAtCMIinQf//AXEiBUUEQCAGQeAAaiAKIAkgCiAJIAlQIgUbeSAFQQZ0rXynIgVBD2sQ9wEgBikDaCEJIAYpA2AhCkEQIAVrIQULIAEgAyAIGyEDIAJC////////P4MhBCAHRQRAIAZB0ABqIAMgBCADIAQgBFAiBxt5IAdBBnStfKciB0EPaxD3AUEQIAdrIQcgBikDWCEEIAYpA1AhAwsgBEIDhiADQj2IhEKAgICAgICABIQhASAJQgOGIApCPYiEIQQgAiALhSENAn4gA0IDhiICIAUgB0YNABogBSAHayIHQf8ASwRAQgAhAUIBDAELIAZBQGsgAiABQYABIAdrEPcBIAZBMGogAiABIAcQ+AEgBikDOCEBIAYpAzAgBikDQCAGKQNIhEIAUq2ECyEJIARCgICAgICAgASEIQwgCkIDhiEKAkAgDUIAUwRAQgAhA0IAIQQgCSAKhSABIAyFhFANAiAKIAl9IQIgDCABfSAJIApWrX0iBEL/////////A1YNASAGQSBqIAIgBCACIAQgBFAiBxt5IAdBBnStfKdBDGsiBxD3ASAFIAdrIQUgBikDKCEEIAYpAyAhAgwBCyAJIAp8IgIgCVStIAEgDHx8IgRCgICAgICAgAiDUA0AIAlCAYMgBEI/hiACQgGIhIQhAiAFQQFqIQUgBEIBiCEECyALQoCAgICAgICAgH+DIQEgBUH//wFOBEAgAUKAgICAgIDA//8AhCEEQgAhAwwBC0EAIQcCQCAFQQBKBEAgBSEHDAELIAZBEGogAiAEIAVB/wBqEPcBIAYgAiAEQQEgBWsQ+AEgBikDACAGKQMQIAYpAxiEQgBSrYQhAiAGKQMIIQQLIAKnQQdxIgVBBEutIARCPYYgAkIDiIQiAnwiAyACVK0gBEIDiEL///////8/gyAHrUIwhoQgAYR8IQQCQCAFQQRGBEAgBCADQgGDIgEgA3wiAyABVK18IQQMAQsgBUUNAQsLIAAgAzcDACAAIAQ3AwggBkHwAGokAAv6AQIDfgJ/IwBBEGsiBSQAAn4gAb0iA0L///////////8AgyICQoCAgICAgIAIfUL/////////7/8AWARAIAJCPIYhBCACQgSIQoCAgICAgICAPHwMAQsgAkKAgICAgICA+P8AWgRAIANCPIYhBCADQgSIQoCAgICAgMD//wCEDAELIAJQBEBCAAwBCyAFIAJCACADp2dBIGogAkIgiKdnIAJCgICAgBBUGyIGQTFqEPcBIAUpAwAhBCAFKQMIQoCAgICAgMAAhUGM+AAgBmutQjCGhAshAiAAIAQ3AwAgACACIANCgICAgICAgICAf4OENwMIIAVBEGokAAvbAQIBfwJ+QQEhBAJAIABCAFIgAUL///////////8AgyIFQoCAgICAgMD//wBWIAVCgICAgICAwP//AFEbDQAgAkIAUiADQv///////////wCDIgZCgICAgICAwP//AFYgBkKAgICAgIDA//8AURsNACAAIAKEIAUgBoSEUARAQQAPCyABIAODQgBZBEBBfyEEIAAgAlQgASADUyABIANRGw0BIAAgAoUgASADhYRCAFIPC0F/IQQgACACViABIANVIAEgA1EbDQAgACAChSABIAOFhEIAUiEECyAEC8ABAgF/An5BfyEDAkAgAEIAUiABQv///////////wCDIgRCgICAgICAwP//AFYgBEKAgICAgIDA//8AURsNACACQv///////////wCDIgVCgICAgICAwP//AFYgBUKAgICAgIDA//8AUnENACAAIAQgBYSEUARAQQAPCyABIAKDQgBZBEAgASACUiABIAJTcQ0BIAAgASAChYRCAFIPCyAAQgBSIAEgAlUgASACURsNACAAIAEgAoWEQgBSIQMLIAMLqQEBAXxEAAAAAAAA8D8hAQJAIABBgAhOBEBEAAAAAAAA4H8hASAAQf8PSQRAIABB/wdrIQAMAgtEAAAAAAAA8H8hAUH9FyAAIABB/RdOG0H+D2shAAwBCyAAQYF4Sg0ARAAAAAAAAGADIQEgAEG4cEsEQCAAQckHaiEADAELRAAAAAAAAAAAIQFB8GggACAAQfBoTBtBkg9qIQALIAEgAEH/B2qtQjSGv6ILNQAgACABNwMAIAAgAkL///////8/gyAEQjCIp0GAgAJxIAJCMIinQf//AXFyrUIwhoQ3AwgLZAIBfwF+IwBBEGsiAiQAIAACfiABRQRAQgAMAQsgAiABrUIAIAFnIgFB0QBqEPcBIAIpAwhCgICAgICAwACFQZ6AASABa61CMIZ8IQMgAikDAAs3AwAgACADNwMIIAJBEGokAAtFAQF/IwBBEGsiBSQAIAUgASACIAMgBEKAgICAgICAgIB/hRDnAiAFKQMAIQEgACAFKQMINwMIIAAgATcDACAFQRBqJAALxAIBAX8jAEHQAGsiBCQAAkAgA0GAgAFOBEAgBEEgaiABIAJCAEKAgICAgICA//8AEOYCIAQpAyghAiAEKQMgIQEgA0H//wFJBEAgA0H//wBrIQMMAgsgBEEQaiABIAJCAEKAgICAgICA//8AEOYCQf3/AiADIANB/f8CThtB/v8BayEDIAQpAxghAiAEKQMQIQEMAQsgA0GBgH9KDQAgBEFAayABIAJCAEKAgICAgICAORDmAiAEKQNIIQIgBCkDQCEBIANB9IB+SwRAIANBjf8AaiEDDAELIARBMGogASACQgBCgICAgICAgDkQ5gJB6IF9IAMgA0HogX1MG0Ga/gFqIQMgBCkDOCECIAQpAzAhAQsgBCABIAJCACADQf//AGqtQjCGEOYCIAAgBCkDCDcDCCAAIAQpAwA3AwAgBEHQAGokAAt1AQF+IAAgASAEfiACIAN+fCADQiCIIgIgAUIgiCIEfnwgA0L/////D4MiAyABQv////8PgyIBfiIFQiCIIAMgBH58IgNCIIh8IAEgAn4gA0L/////D4N8IgFCIIh8NwMIIAAgBUL/////D4MgAUIghoQ3AwALvg8CBX8PfiMAQdACayIFJAAgBEL///////8/gyELIAJC////////P4MhCiACIASFQoCAgICAgICAgH+DIQ0gBEIwiKdB//8BcSEIAkACQCACQjCIp0H//wFxIglB//8Ba0GCgH5PBEAgCEH//wFrQYGAfksNAQsgAVAgAkL///////////8AgyIMQoCAgICAgMD//wBUIAxCgICAgICAwP//AFEbRQRAIAJCgICAgICAIIQhDQwCCyADUCAEQv///////////wCDIgJCgICAgICAwP//AFQgAkKAgICAgIDA//8AURtFBEAgBEKAgICAgIAghCENIAMhAQwCCyABIAxCgICAgICAwP//AIWEUARAIAMgAkKAgICAgIDA//8AhYRQBEBCACEBQoCAgICAgOD//wAhDQwDCyANQoCAgICAgMD//wCEIQ1CACEBDAILIAMgAkKAgICAgIDA//8AhYRQBEBCACEBDAILIAEgDIRQBEBCgICAgICA4P//ACANIAIgA4RQGyENQgAhAQwCCyACIAOEUARAIA1CgICAgICAwP//AIQhDUIAIQEMAgsgDEL///////8/WARAIAVBwAJqIAEgCiABIAogClAiBht5IAZBBnStfKciBkEPaxD3AUEQIAZrIQYgBSkDyAIhCiAFKQPAAiEBCyACQv///////z9WDQAgBUGwAmogAyALIAMgCyALUCIHG3kgB0EGdK18pyIHQQ9rEPcBIAYgB2pBEGshBiAFKQO4AiELIAUpA7ACIQMLIAVBoAJqIAtCgICAgICAwACEIhJCD4YgA0IxiIQiAkIAQoCAgICw5ryC9QAgAn0iBEIAEPACIAVBkAJqQgAgBSkDqAJ9QgAgBEIAEPACIAVBgAJqIAUpA5gCQgGGIAUpA5ACQj+IhCIEQgAgAkIAEPACIAVB8AFqIARCAEIAIAUpA4gCfUIAEPACIAVB4AFqIAUpA/gBQgGGIAUpA/ABQj+IhCIEQgAgAkIAEPACIAVB0AFqIARCAEIAIAUpA+gBfUIAEPACIAVBwAFqIAUpA9gBQgGGIAUpA9ABQj+IhCIEQgAgAkIAEPACIAVBsAFqIARCAEIAIAUpA8gBfUIAEPACIAVBoAFqIAJCACAFKQO4AUIBhiAFKQOwAUI/iIRCAX0iAkIAEPACIAVBkAFqIANCD4ZCACACQgAQ8AIgBUHwAGogAkIAQgAgBSkDqAEgBSkDoAEiDCAFKQOYAXwiBCAMVK18IARCAVatfH1CABDwAiAFQYABakIBIAR9QgAgAkIAEPACIAYgCSAIa2ohBgJ/IAUpA3AiE0IBhiIOIAUpA4gBIg9CAYYgBSkDgAFCP4iEfCIQQufsAH0iFEIgiCICIApCgICAgICAwACEIhVCAYYiFkIgiCIEfiIRIAFCAYYiDEIgiCILIBAgFFatIA4gEFatIAUpA3hCAYYgE0I/iIQgD0I/iHx8fEIBfSITQiCIIhB+fCIOIBFUrSAOIA4gE0L/////D4MiEyABQj+IIhcgCkIBhoRC/////w+DIgp+fCIOVq18IAQgEH58IAQgE34iESAKIBB+fCIPIBFUrUIghiAPQiCIhHwgDiAOIA9CIIZ8Ig5WrXwgDiAOIBRC/////w+DIhQgCn4iESACIAt+fCIPIBFUrSAPIA8gEyAMQv7///8PgyIRfnwiD1atfHwiDlatfCAOIAQgFH4iGCAQIBF+fCIEIAIgCn58IgogCyATfnwiEEIgiCAKIBBWrSAEIBhUrSAEIApWrXx8QiCGhHwiBCAOVK18IAQgDyACIBF+IgIgCyAUfnwiC0IgiCACIAtWrUIghoR8IgIgD1StIAIgEEIghnwgAlStfHwiAiAEVK18IgRC/////////wBYBEAgFiAXhCEVIAVB0ABqIAIgBCADIBIQ8AIgAUIxhiAFKQNYfSAFKQNQIgFCAFKtfSEKQgAgAX0hCyAGQf7/AGoMAQsgBUHgAGogBEI/hiACQgGIhCICIARCAYgiBCADIBIQ8AIgAUIwhiAFKQNofSAFKQNgIgxCAFKtfSEKQgAgDH0hCyABIQwgBkH//wBqCyIGQf//AU4EQCANQoCAgICAgMD//wCEIQ1CACEBDAELAn4gBkEASgRAIApCAYYgC0I/iIQhCiAEQv///////z+DIAatQjCGhCEMIAtCAYYMAQsgBkGPf0wEQEIAIQEMAgsgBUFAayACIARBASAGaxD4ASAFQTBqIAwgFSAGQfAAahD3ASAFQSBqIAMgEiAFKQNAIgIgBSkDSCIMEPACIAUpAzggBSkDKEIBhiAFKQMgIgFCP4iEfSAFKQMwIgQgAUIBhiIBVK19IQogBCABfQshBCAFQRBqIAMgEkIDQgAQ8AIgBSADIBJCBUIAEPACIAwgAiACIAMgAkIBgyIBIAR8IgNUIAogASADVq18IgEgElYgASASURutfCICVq18IgQgAiACIARCgICAgICAwP//AFQgAyAFKQMQViABIAUpAxgiBFYgASAEURtxrXwiAlatfCIEIAIgBEKAgICAgIDA//8AVCADIAUpAwBWIAEgBSkDCCIDViABIANRG3GtfCIBIAJUrXwgDYQhDQsgACABNwMAIAAgDTcDCCAFQdACaiQAC9EGAgR/A34jAEGAAWsiBSQAAkACQAJAIAMgBEIAQgAQ6QJFDQACfyAEQv///////z+DIQoCfyAEQjCIp0H//wFxIgZB//8BRwRAQQQgBg0BGkECQQMgAyAKhFAbDAILIAMgCoRQCwshBiACQjCIpyIIQf//AXEiB0H//wFGDQAgBg0BCyAFQRBqIAEgAiADIAQQ5gIgBSAFKQMQIgIgBSkDGCIBIAIgARDxAiAFKQMIIQIgBSkDACEEDAELIAEgAkL///////////8AgyIKIAMgBEL///////////8AgyIJEOkCQQBMBEAgASAKIAMgCRDpAgRAIAEhBAwCCyAFQfAAaiABIAJCAEIAEOYCIAUpA3ghAiAFKQNwIQQMAQsgBEIwiKdB//8BcSEGIAcEfiABBSAFQeAAaiABIApCAEKAgICAgIDAu8AAEOYCIAUpA2giCkIwiKdB+ABrIQcgBSkDYAshBCAGRQRAIAVB0ABqIAMgCUIAQoCAgICAgMC7wAAQ5gIgBSkDWCIJQjCIp0H4AGshBiAFKQNQIQMLIAlC////////P4NCgICAgICAwACEIQsgCkL///////8/g0KAgICAgIDAAIQhCiAGIAdIBEADQAJ+IAogC30gAyAEVq19IglCAFkEQCAJIAQgA30iBIRQBEAgBUEgaiABIAJCAEIAEOYCIAUpAyghAiAFKQMgIQQMBQsgCUIBhiAEQj+IhAwBCyAKQgGGIARCP4iECyEKIARCAYYhBCAHQQFrIgcgBkoNAAsgBiEHCwJAIAogC30gAyAEVq19IglCAFMEQCAKIQkMAQsgCSAEIAN9IgSEQgBSDQAgBUEwaiABIAJCAEIAEOYCIAUpAzghAiAFKQMwIQQMAQsgCUL///////8/WARAA0AgBEI/iCEBIAdBAWshByAEQgGGIQQgASAJQgGGhCIJQoCAgICAgMAAVA0ACwsgCEGAgAJxIQYgB0EATARAIAVBQGsgBCAJQv///////z+DIAdB+ABqIAZyrUIwhoRCAEKAgICAgIDAwz8Q5gIgBSkDSCECIAUpA0AhBAwBCyAJQv///////z+DIAYgB3KtQjCGhCECCyAAIAQ3AwAgACACNwMIIAVBgAFqJAALizMDD38HfgF8IwBBMGsiDCQAAkAgAkECTQRAIAJBAnQiAkHsmgFqKAIAIQ8gAkHgmgFqKAIAIQ4DQAJ/IAEoAgQiAiABKAJoRwRAIAEgAkEBajYCBCACLQAADAELIAEQ5AILIgJBIEYgAkEJa0EFSXINAAtBASEGAkACQCACQStrDgMAAQABC0F/QQEgAkEtRhshBiABKAIEIgIgASgCaEcEQCABIAJBAWo2AgQgAi0AACECDAELIAEQ5AIhAgsCQAJAA0AgBUHRCWosAAAgAkEgckYEQAJAIAVBBksNACABKAIEIgIgASgCaEcEQCABIAJBAWo2AgQgAi0AACECDAELIAEQ5AIhAgsgBUEBaiIFQQhHDQEMAgsLIAVBA0cEQCAFQQhGDQEgA0UNAiAFQQRJDQIgBUEIRg0BCyABKQNwIhNCAFkEQCABIAEoAgRBAWs2AgQLIANFDQAgBUEESQ0AIBNCAFMhAgNAIAJFBEAgASABKAIEQQFrNgIECyAFQQFrIgVBA0sNAAsLQgAhEyMAQRBrIgIkAAJ+IAayQwAAgH+UvCIDQf////8HcSIBQYCAgARrQf////cHTQRAIAGtQhmGQoCAgICAgIDAP3wMAQsgA61CGYZCgICAgICAwP//AIQgAUGAgID8B08NABpCACABRQ0AGiACIAGtQgAgAWciAUHRAGoQ9wEgAikDACETIAIpAwhCgICAgICAwACFQYn/ACABa61CMIaECyEUIAwgEzcDACAMIBQgA0GAgICAeHGtQiCGhDcDCCACQRBqJAAgDCkDCCETIAwpAwAhFAwCCwJAAkACQCAFDQBBACEFA0AgBUGuHGosAAAgAkEgckcNAQJAIAVBAUsNACABKAIEIgIgASgCaEcEQCABIAJBAWo2AgQgAi0AACECDAELIAEQ5AIhAgsgBUEBaiIFQQNHDQALDAELAkACQCAFDgQAAQECAQsCQCACQTBHDQACfyABKAIEIgUgASgCaEcEQCABIAVBAWo2AgQgBS0AAAwBCyABEOQCC0FfcUHYAEYEQCMAQbADayICJAACfyABKAIEIgUgASgCaEcEQCABIAVBAWo2AgQgBS0AAAwBCyABEOQCCyEFAkACfwNAAkAgBUEwRwRAIAVBLkcNBCABKAIEIgUgASgCaEYNASABIAVBAWo2AgQgBS0AAAwDCyABKAIEIgUgASgCaEcEQEEBIQkgASAFQQFqNgIEIAUtAAAhBQwCC0EBIQkgARDkAiEFDAELCyABEOQCCyEFQQEhBCAFQTBHDQADQAJ/IAEoAgQiBSABKAJoRwRAIAEgBUEBajYCBCAFLQAADAELIAEQ5AILIQUgFkIBfSEWIAVBMEYNAAtBASEJC0KAgICAgIDA/z8hFANAAkAgBUEgciELAkACQCAFQTBrIghBCkkNACAFQS5HIAtB4QBrQQZPcQ0CIAVBLkcNACAEDQJBASEEIBMhFgwBCyALQdcAayAIIAVBOUobIQUCQCATQgdXBEAgBSAKQQR0aiEKDAELIBNCHFgEQCACQTBqIAUQ5QIgAkEgaiAYIBRCAEKAgICAgIDA/T8Q5gIgAkEQaiACKQMwIAIpAzggAikDICIYIAIpAygiFBDmAiACIAIpAxAgAikDGCAVIBcQ5wIgAikDCCEXIAIpAwAhFQwBCyAFRQ0AIAcNACACQdAAaiAYIBRCAEKAgICAgICA/z8Q5gIgAkFAayACKQNQIAIpA1ggFSAXEOcCIAIpA0ghF0EBIQcgAikDQCEVCyATQgF8IRNBASEJCyABKAIEIgUgASgCaEcEQCABIAVBAWo2AgQgBS0AACEFDAILIAEQ5AIhBQwBCwsCfiAJRQRAAkACQCABKQNwQgBZBEAgASABKAIEIgVBAWs2AgQgA0UNASABIAVBAms2AgQgBEUNAiABIAVBA2s2AgQMAgsgAw0BCyABQgAQ4wILIAJB4ABqIAa3RAAAAAAAAAAAohDoAiACKQNgIRUgAikDaAwBCyATQgdXBEAgEyEUA0AgCkEEdCEKIBRCAXwiFEIIUg0ACwsCQAJAAkAgBUFfcUHQAEYEQCABIAMQ9AIiFEKAgICAgICAgIB/Ug0DIAMEQCABKQNwQgBZDQIMAwtCACEVIAFCABDjAkIADAQLQgAhFCABKQNwQgBTDQILIAEgASgCBEEBazYCBAtCACEUCyAKRQRAIAJB8ABqIAa3RAAAAAAAAAAAohDoAiACKQNwIRUgAikDeAwBCyAWIBMgBBtCAoYgFHxCIH0iE0EAIA9rrVUEQEG47AJBxAA2AgAgAkGgAWogBhDlAiACQZABaiACKQOgASACKQOoAUJ/Qv///////7///wAQ5gIgAkGAAWogAikDkAEgAikDmAFCf0L///////+///8AEOYCIAIpA4ABIRUgAikDiAEMAQsgD0HiAWusIBNXBEAgCkEATgRAA0AgAkGgA2ogFSAXQgBCgICAgICAwP+/fxDnAiAVIBdCgICAgICAgP8/EOoCIQEgAkGQA2ogFSAXIAIpA6ADIBUgAUEATiIBGyACKQOoAyAXIAEbEOcCIBNCAX0hEyACKQOYAyEXIAIpA5ADIRUgCkEBdCABciIKQQBODQALCwJ+IBMgD6x9QiB8IhSnIgFBACABQQBKGyAOIBQgDq1TGyIBQfEATgRAIAJBgANqIAYQ5QIgAikDiAMhFiACKQOAAyEYQgAMAQsgAkHgAmpBkAEgAWsQ6wIQ6AIgAkHQAmogBhDlAiACQfACaiACKQPgAiACKQPoAiACKQPQAiIYIAIpA9gCIhYQ7AIgAikD+AIhGSACKQPwAgshFCACQcACaiAKIApBAXFFIBUgF0IAQgAQ6QJBAEcgAUEgSHFxIgFqEO0CIAJBsAJqIBggFiACKQPAAiACKQPIAhDmAiACQZACaiACKQOwAiACKQO4AiAUIBkQ5wIgAkGgAmogGCAWQgAgFSABG0IAIBcgARsQ5gIgAkGAAmogAikDoAIgAikDqAIgAikDkAIgAikDmAIQ5wIgAkHwAWogAikDgAIgAikDiAIgFCAZEO4CIAIpA/ABIhQgAikD+AEiFkIAQgAQ6QJFBEBBuOwCQcQANgIACyACQeABaiAUIBYgE6cQ7wIgAikD4AEhFSACKQPoAQwBC0G47AJBxAA2AgAgAkHQAWogBhDlAiACQcABaiACKQPQASACKQPYAUIAQoCAgICAgMAAEOYCIAJBsAFqIAIpA8ABIAIpA8gBQgBCgICAgICAwAAQ5gIgAikDsAEhFSACKQO4AQshEyAMIBU3AxAgDCATNwMYIAJBsANqJAAgDCkDGCETIAwpAxAhFAwGCyABKQNwQgBTDQAgASABKAIEQQFrNgIECyABIQUgBiEKIAMhCUEAIQFBACEGIwBBkMYAayIEJABBACAPayIQIA5rIRICQAJ/A0ACQCACQTBHBEAgAkEuRw0EIAUoAgQiAiAFKAJoRg0BIAUgAkEBajYCBCACLQAADAMLIAUoAgQiAiAFKAJoRwRAQQEhASAFIAJBAWo2AgQgAi0AACECDAILQQEhASAFEOQCIQIMAQsLIAUQ5AILIQJBASEHIAJBMEcNAANAAn8gBSgCBCIBIAUoAmhHBEAgBSABQQFqNgIEIAEtAAAMAQsgBRDkAgshAiATQgF9IRMgAkEwRg0AC0EBIQELIARBADYCkAYgAkEwayEIIAwCfgJAAkACQAJAAkACQCACQS5GIgMNACAIQQlNDQAMAQsDQAJAIANBAXEEQCAHRQRAIBQhE0EBIQcMAgsgAUUhAwwECyAUQgF8IRQgBkH8D0wEQCANIBSnIAJBMEYbIQ0gBEGQBmogBkECdGoiASALBH8gAiABKAIAQQpsakEwawUgCAs2AgBBASEBQQAgC0EBaiICIAJBCUYiAhshCyACIAZqIQYMAQsgAkEwRg0AIAQgBCgCgEZBAXI2AoBGQdyPASENCwJ/IAUoAgQiAiAFKAJoRwRAIAUgAkEBajYCBCACLQAADAELIAUQ5AILIgJBMGshCCACQS5GIgMNACAIQQpJDQALCyATIBQgBxshEwJAIAFFDQAgAkFfcUHFAEcNAAJAIAUgCRD0AiIVQoCAgICAgICAgH9SDQAgCUUNBEIAIRUgBSkDcEIAUw0AIAUgBSgCBEEBazYCBAsgEyAVfCETDAQLIAFFIQMgAkEASA0BCyAFKQNwQgBTDQAgBSAFKAIEQQFrNgIECyADRQ0BQbjsAkEcNgIAC0IAIRQgBUIAEOMCQgAMAQsgBCgCkAYiAUUEQCAEIAq3RAAAAAAAAAAAohDoAiAEKQMAIRQgBCkDCAwBCwJAIBRCCVUNACATIBRSDQAgDkEeTEEAIAEgDnYbDQAgBEEwaiAKEOUCIARBIGogARDtAiAEQRBqIAQpAzAgBCkDOCAEKQMgIAQpAygQ5gIgBCkDECEUIAQpAxgMAQsgEEEBdq0gE1MEQEG47AJBxAA2AgAgBEHgAGogChDlAiAEQdAAaiAEKQNgIAQpA2hCf0L///////+///8AEOYCIARBQGsgBCkDUCAEKQNYQn9C////////v///ABDmAiAEKQNAIRQgBCkDSAwBCyAPQeIBa6wgE1UEQEG47AJBxAA2AgAgBEGQAWogChDlAiAEQYABaiAEKQOQASAEKQOYAUIAQoCAgICAgMAAEOYCIARB8ABqIAQpA4ABIAQpA4gBQgBCgICAgICAwAAQ5gIgBCkDcCEUIAQpA3gMAQsgCwRAIAtBCEwEQCAEQZAGaiAGQQJ0aiIBKAIAIQUDQCAFQQpsIQUgC0EBaiILQQlHDQALIAEgBTYCAAsgBkEBaiEGCyATpyEHAkAgDUEJTg0AIAcgDUgNACAHQRFKDQAgB0EJRgRAIARBwAFqIAoQ5QIgBEGwAWogBCgCkAYQ7QIgBEGgAWogBCkDwAEgBCkDyAEgBCkDsAEgBCkDuAEQ5gIgBCkDoAEhFCAEKQOoAQwCCyAHQQhMBEAgBEGQAmogChDlAiAEQYACaiAEKAKQBhDtAiAEQfABaiAEKQOQAiAEKQOYAiAEKQOAAiAEKQOIAhDmAiAEQeABakEAIAdrQQJ0QeCaAWooAgAQ5QIgBEHQAWogBCkD8AEgBCkD+AEgBCkD4AEgBCkD6AEQ8QIgBCkD0AEhFCAEKQPYAQwCCyAOIAdBfWxqQRtqIgFBHkxBACAEKAKQBiICIAF2Gw0AIARB4AJqIAoQ5QIgBEHQAmogAhDtAiAEQcACaiAEKQPgAiAEKQPoAiAEKQPQAiAEKQPYAhDmAiAEQbACaiAHQQJ0QZiaAWooAgAQ5QIgBEGgAmogBCkDwAIgBCkDyAIgBCkDsAIgBCkDuAIQ5gIgBCkDoAIhFCAEKQOoAgwBCwNAIARBkAZqIAYiAkEBayIGQQJ0aigCAEUNAAtBACELAkAgB0EJbyIBRQRAQQAhAwwBC0EAIQMgAUEJaiABIAdBAEgbIQECQCACRQRAQQAhAgwBC0GAlOvcA0EAIAFrQQJ0QeCaAWooAgAiBm0hCUEAIQhBACEFA0AgBEGQBmogBUECdGoiDSAIIA0oAgAiDSAGbiIQaiIINgIAIANBAWpB/w9xIAMgCEUgAyAFRnEiCBshAyAHQQlrIAcgCBshByAJIA0gBiAQbGtsIQggBUEBaiIFIAJHDQALIAhFDQAgBEGQBmogAkECdGogCDYCACACQQFqIQILIAcgAWtBCWohBwsDQCAEQZAGaiADQQJ0aiEJAkADQCAHQSROBEAgB0EkRw0CIAkoAgBB0en5BE8NAgsgAkH/D2ohBkEAIQggAiEBA0AgASECIAitIARBkAZqIAZB/w9xIgVBAnRqIgE1AgBCHYZ8IhNCgZTr3ANUBH9BAAUgEyATQoCU69wDgCIUQoCU69wDfn0hEyAUpwshCCABIBOnIgE2AgAgAiACIAIgBSABGyADIAVGGyAFIAJBAWtB/w9xRxshASAFQQFrIQYgAyAFRw0ACyALQR1rIQsgCEUNAAsgASADQQFrQf8PcSIDRgRAIARBkAZqIgYgAUH+D2pB/w9xQQJ0aiICIAIoAgAgBiABQQFrQf8PcSICQQJ0aigCAHI2AgALIAdBCWohByAEQZAGaiADQQJ0aiAINgIADAELCwJAA0AgAkEBakH/D3EhBiAEQZAGaiACQQFrQf8PcUECdGohCANAQQlBASAHQS1KGyEJAkADQCADIQFBACEFAkADQAJAIAEgBWpB/w9xIgMgAkYNACAEQZAGaiADQQJ0aigCACIDIAVBAnRBsJoBaigCACINSQ0AIAMgDUsNAiAFQQFqIgVBBEcNAQsLIAdBJEcNAEIAIRNBACEFQgAhFANAIAIgASAFakH/D3EiA0YEQCACQQFqQf8PcSICQQJ0IARqQQA2AowGCyAEQYAGaiAEQZAGaiADQQJ0aigCABDtAiAEQfAFaiATIBRCAEKAgICA5Zq3jsAAEOYCIARB4AVqIAQpA/AFIAQpA/gFIAQpA4AGIAQpA4gGEOcCIAQpA+gFIRQgBCkD4AUhEyAFQQFqIgVBBEcNAAsgBEHQBWogChDlAiAEQcAFaiATIBQgBCkD0AUgBCkD2AUQ5gIgBCkDyAUhFEIAIRMgBCkDwAUhFSALQfEAaiIHIA9rIgZBACAGQQBKGyAOIAYgDkgiBRsiA0HwAEwNAgwFCyAJIAtqIQsgAiEDIAEgAkYNAAtBgJTr3AMgCXYhDUF/IAl0QX9zIRBBACEFIAEhAwNAIARBkAZqIAFBAnRqIhEgBSARKAIAIhEgCXZqIgU2AgAgA0EBakH/D3EgAyAFRSABIANGcSIFGyEDIAdBCWsgByAFGyEHIBAgEXEgDWwhBSABQQFqQf8PcSIBIAJHDQALIAVFDQEgAyAGRwRAIARBkAZqIAJBAnRqIAU2AgAgBiECDAMLIAggCCgCAEEBcjYCAAwBCwsLIARBkAVqQeEBIANrEOsCEOgCIARBsAVqIAQpA5AFIAQpA5gFIBUgFBDsAiAEKQO4BSEYIAQpA7AFIRcgBEGABWpB8QAgA2sQ6wIQ6AIgBEGgBWogFSAUIAQpA4AFIAQpA4gFEPICIARB8ARqIBUgFCAEKQOgBSITIAQpA6gFIhYQ7gIgBEHgBGogFyAYIAQpA/AEIAQpA/gEEOcCIAQpA+gEIRQgBCkD4AQhFQsCQCABQQRqQf8PcSIJIAJGDQACQCAEQZAGaiAJQQJ0aigCACIJQf/Jte4BTQRAIAlFIAFBBWpB/w9xIAJGcQ0BIARB8ANqIAq3RAAAAAAAANA/ohDoAiAEQeADaiATIBYgBCkD8AMgBCkD+AMQ5wIgBCkD6AMhFiAEKQPgAyETDAELIAlBgMq17gFHBEAgBEHQBGogCrdEAAAAAAAA6D+iEOgCIARBwARqIBMgFiAEKQPQBCAEKQPYBBDnAiAEKQPIBCEWIAQpA8AEIRMMAQsgCrchGiACIAFBBWpB/w9xRgRAIARBkARqIBpEAAAAAAAA4D+iEOgCIARBgARqIBMgFiAEKQOQBCAEKQOYBBDnAiAEKQOIBCEWIAQpA4AEIRMMAQsgBEGwBGogGkQAAAAAAADoP6IQ6AIgBEGgBGogEyAWIAQpA7AEIAQpA7gEEOcCIAQpA6gEIRYgBCkDoAQhEwsgA0HvAEoNACAEQdADaiATIBZCAEKAgICAgIDA/z8Q8gIgBCkD0AMgBCkD2ANCAEIAEOkCDQAgBEHAA2ogEyAWQgBCgICAgICAwP8/EOcCIAQpA8gDIRYgBCkDwAMhEwsgBEGwA2ogFSAUIBMgFhDnAiAEQaADaiAEKQOwAyAEKQO4AyAXIBgQ7gIgBCkDqAMhFCAEKQOgAyEVAkAgEkECayAHQf////8HcU4NACAEIBRC////////////AIM3A5gDIAQgFTcDkAMgBEGAA2ogFSAUQgBCgICAgICAgP8/EOYCIAQpA5ADIAQpA5gDQoCAgICAgIC4wAAQ6gIhASAEKQOIAyAUIAFBAE4iARshFCAEKQOAAyAVIAEbIRUgEyAWQgBCABDpAkEARyAFIAMgBkdxIAUgARtxIBIgASALaiILQe4AakhyRQ0AQbjsAkHEADYCAAsgBEHwAmogFSAUIAsQ7wIgBCkD8AIhFCAEKQP4Ags3AyggDCAUNwMgIARBkMYAaiQAIAwpAyghEyAMKQMgIRQMBAsgASkDcEIAWQRAIAEgASgCBEEBazYCBAsMAQsCQAJ/IAEoAgQiAiABKAJoRwRAIAEgAkEBajYCBCACLQAADAELIAEQ5AILQShGBEBBASEFDAELQoCAgICAgOD//wAhEyABKQNwQgBTDQMgASABKAIEQQFrNgIEDAMLA0ACfyABKAIEIgIgASgCaEcEQCABIAJBAWo2AgQgAi0AAAwBCyABEOQCCyICQcEAayEGAkACQCACQTBrQQpJDQAgBkEaSQ0AIAJB3wBGDQAgAkHhAGtBGk8NAQsgBUEBaiEFDAELC0KAgICAgIDg//8AIRMgAkEpRg0CIAEpA3AiFkIAWQRAIAEgASgCBEEBazYCBAsCQCADBEAgBQ0BDAQLDAELA0AgBUEBayEFIBZCAFkEQCABIAEoAgRBAWs2AgQLIAUNAAsMAgtBuOwCQRw2AgAgAUIAEOMCC0IAIRMLIAAgFDcDACAAIBM3AwggDEEwaiQAC5kEAgR/AX4CQAJAAkACQAJAAn8gACgCBCICIAAoAmhHBEAgACACQQFqNgIEIAItAAAMAQsgABDkAgsiAkEraw4DAAEAAQsCfyAAKAIEIgMgACgCaEcEQCAAIANBAWo2AgQgAy0AAAwBCyAAEOQCCyEDIAJBLUYhBSADQTprIQQgAUUNASAEQXVLDQEgACkDcEIAUw0CIAAgACgCBEEBazYCBAwCCyACQTprIQQgAiEDCyAEQXZJDQAgA0EwayIEQQpJBEBBACECA0AgAyACQQpsaiEBAn8gACgCBCICIAAoAmhHBEAgACACQQFqNgIEIAItAAAMAQsgABDkAgsiA0EwayIEQQlNIAFBMGsiAkHMmbPmAEhxDQALIAKsIQYLAkAgBEEKTw0AA0AgA60gBkIKfnwhBgJ/IAAoAgQiASAAKAJoRwRAIAAgAUEBajYCBCABLQAADAELIAAQ5AILIQMgBkIwfSEGIANBMGsiBEEJSw0BIAZCro+F18fC66MBUw0ACwsgBEEKSQRAA0ACfyAAKAIEIgEgACgCaEcEQCAAIAFBAWo2AgQgAS0AAAwBCyAAEOQCC0Ewa0EKSQ0ACwsgACkDcEIAWQRAIAAgACgCBEEBazYCBAtCACAGfSAGIAUbIQYMAQtCgICAgICAgICAfyEGIAApA3BCAFMNACAAIAAoAgRBAWs2AgRCgICAgICAgICAfw8LIAYLtgMCA38BfiMAQSBrIgMkAAJAIAFC////////////AIMiBUKAgICAgIDAwD99IAVCgICAgICAwL/AAH1UBEAgAUIZiKchBCAAUCABQv///w+DIgVCgICACFQgBUKAgIAIURtFBEAgBEGBgICABGohAgwCCyAEQYCAgIAEaiECIAAgBUKAgIAIhYRCAFINASACIARBAXFqIQIMAQsgAFAgBUKAgICAgIDA//8AVCAFQoCAgICAgMD//wBRG0UEQCABQhmIp0H///8BcUGAgID+B3IhAgwBC0GAgID8ByECIAVC////////v7/AAFYNAEEAIQIgBUIwiKciBEGR/gBJDQAgA0EQaiAAIAFC////////P4NCgICAgICAwACEIgUgBEGB/gBrEPcBIAMgACAFQYH/ACAEaxD4ASADKQMIIgBCGYinIQIgAykDACADKQMQIAMpAxiEQgBSrYQiBVAgAEL///8PgyIAQoCAgAhUIABCgICACFEbRQRAIAJBAWohAgwBCyAFIABCgICACIWEQgBSDQAgAkEBcSACaiECCyADQSBqJAAgAiABQiCIp0GAgICAeHFyvgvFAgEEfyADQezxAiADGyIFKAIAIQMCQAJ/AkAgAUUEQCADDQFBAA8LQX4gAkUNARoCQCADBEAgAiEEDAELIAEtAAAiA8AiBEEATgRAIAAEQCAAIAM2AgALIARBAEcPC0HU7QIoAgAoAgBFBEBBASAARQ0DGiAAIAEsAABB/78DcTYCAEEBDwsgAS0AAEHCAWsiA0EySw0BIANBAnRBkJ0BaigCACEDIAJBAWsiBEUNAyABQQFqIQELIAEtAAAiBkEDdiIHQRBrIANBGnUgB2pyQQdLDQADQCAEQQFrIQQgBkGAAWsgA0EGdHIiA0EATgRAIAVBADYCACAABEAgACADNgIACyACIARrDwsgBEUNAyABQQFqIgEtAAAiBkHAAXFBgAFGDQALCyAFQQA2AgBBuOwCQRk2AgBBfwsPCyAFIAM2AgBBfgtDAAJAIABFDQACQAJAAkACQCABQQJqDgYAAQICBAMECyAAIAI8AAAPCyAAIAI9AQAPCyAAIAI+AgAPCyAAIAI3AwALC9wfAg9/Bn4jAEGQAWsiCCQAIAhBAEGQARDUASIIQX82AkwgCCAANgIsIAhBwgE2AiAgCCAANgJUIAEhBCACIQ9BACEAIwBBsAJrIgckACAIIgMoAkwaAkACQAJAAkAgAygCBA0AIAMQhwIaIAMoAgQNAAwBCyAELQAAIgFFDQICQAJAAkACQANAAkACQCABQf8BcSIBQSBGIAFBCWtBBUlyBEADQCAEIgFBAWohBCABLQABIgJBIEYgAkEJa0EFSXINAAsgA0IAEOMCA0ACfyADKAIEIgIgAygCaEcEQCADIAJBAWo2AgQgAi0AAAwBCyADEOQCCyICQSBGIAJBCWtBBUlyDQALIAMoAgQhBCADKQNwQgBZBEAgAyAEQQFrIgQ2AgQLIAQgAygCLGusIAMpA3ggFXx8IRUMAQsCfwJAAkAgBC0AAEElRgRAIAQtAAEiAUEqRg0BIAFBJUcNAgsgA0IAEOMCAkAgBC0AAEElRgRAA0ACfyADKAIEIgEgAygCaEcEQCADIAFBAWo2AgQgAS0AAAwBCyADEOQCCyIBQSBGIAFBCWtBBUlyDQALIARBAWohBAwBCyADKAIEIgEgAygCaEcEQCADIAFBAWo2AgQgAS0AACEBDAELIAMQ5AIhAQsgBC0AACABRwRAIAMpA3BCAFkEQCADIAMoAgRBAWs2AgQLIAFBAE4NDUEAIQYgDQ0NDAsLIAMoAgQgAygCLGusIAMpA3ggFXx8IRUgBCEBDAMLQQAhCSAEQQJqDAELAkAgAUEwa0EKTw0AIAQtAAJBJEcNACAELQABQTBrIQgjAEEQayICIA82AgwgAiAPIAhBAnRBBGtBACAIQQFLG2oiAkEEajYCCCACKAIAIQkgBEEDagwBCyAPKAIAIQkgD0EEaiEPIARBAWoLIQFBACEIQQAhBCABLQAAQTBrQQpJBEADQCABLQAAIARBCmxqQTBrIQQgAS0AASECIAFBAWohASACQTBrQQpJDQALCyABLQAAIg5B7QBHBH8gAQVBACEKIAlBAEchCCABLQABIQ5BACEAIAFBAWoLIgJBAWohAUEDIQUgCCEGAkACQAJAAkACQAJAIA5BwQBrDjoEDAQMBAQEDAwMDAMMDAwMDAwEDAwMDAQMDAQMDAwMDAQMBAQEBAQABAUMAQwEBAQMDAQCBAwMBAwCDAsgAkECaiABIAItAAFB6ABGIgIbIQFBfkF/IAIbIQUMBAsgAkECaiABIAItAAFB7ABGIgIbIQFBA0EBIAIbIQUMAwtBASEFDAILQQIhBQwBC0EAIQUgAiEBC0EBIAUgAS0AACIGQS9xQQNGIgIbIRACQCAGQSByIAYgAhsiC0HbAEYNAAJAIAtB7gBHBEAgC0HjAEcNAUEBIAQgBEEBTBshBAwCCyAJIBAgFRD3AgwCCyADQgAQ4wIDQAJ/IAMoAgQiAiADKAJoRwRAIAMgAkEBajYCBCACLQAADAELIAMQ5AILIgJBIEYgAkEJa0EFSXINAAsgAygCBCECIAMpA3BCAFkEQCADIAJBAWsiAjYCBAsgAiADKAIsa6wgAykDeCAVfHwhFQsgAyAErCIUEOMCAkAgAygCBCICIAMoAmhHBEAgAyACQQFqNgIEDAELIAMQ5AJBAEgNBgsgAykDcEIAWQRAIAMgAygCBEEBazYCBAtBECECAkACQAJAAkACQAJAAkACQAJAAkAgC0HYAGsOIQYJCQIJCQkJCQEJAgQBAQEJBQkJCQkJAwYJCQIJBAkJBgALIAtBwQBrIgJBBksNCEEBIAJ0QfEAcUUNCAsgB0EIaiADIBBBABDzAiADKQN4QgAgAygCBCADKAIsa6x9Ug0FDAwLIAtBEHJB8wBGBEAgB0EgakF/QYECENQBGiAHQQA6ACAgC0HzAEcNBiAHQQA6AEEgB0EAOgAuIAdBADYBKgwGCyAHQSBqIAEtAAEiBUHeAEYiBkGBAhDUARogB0EAOgAgIAFBAmogAUEBaiAGGyECAn8CQAJAIAFBAkEBIAYbai0AACIBQS1HBEAgAUHdAEYNASAFQd4ARyEFIAIMAwsgByAFQd4ARyIFOgBODAELIAcgBUHeAEciBToAfgsgAkEBagshAQNAAkAgAS0AACICQS1HBEAgAkUNDyACQd0ARg0IDAELQS0hAiABLQABIgxFDQAgDEHdAEYNACABQQFqIQYCQCAMIAFBAWstAAAiAU0EQCAMIQIMAQsDQCABQQFqIgEgB0EgamogBToAACABIAYtAAAiAkkNAAsLIAYhAQsgAiAHaiAFOgAhIAFBAWohAQwACwALQQghAgwCC0EKIQIMAQtBACECC0IAIRJBACEFQQAhBkEAIQ4jAEEQayIRJAACQCACQQFHIAJBJE1xRQRAQbjsAkEcNgIADAELA0ACfyADKAIEIgQgAygCaEcEQCADIARBAWo2AgQgBC0AAAwBCyADEOQCCyIEQSBGIARBCWtBBUlyDQALAkACQCAEQStrDgMAAQABC0F/QQAgBEEtRhshDiADKAIEIgQgAygCaEcEQCADIARBAWo2AgQgBC0AACEEDAELIAMQ5AIhBAsCQAJAAkACQAJAIAJBAEcgAkEQR3ENACAEQTBHDQACfyADKAIEIgQgAygCaEcEQCADIARBAWo2AgQgBC0AAAwBCyADEOQCCyIEQV9xQdgARgRAAn8gAygCBCICIAMoAmhHBEAgAyACQQFqNgIEIAItAAAMAQsgAxDkAgshBEEQIQIgBEGBmwFqLQAAQRBJDQMgAykDcEIAWQRAIAMgAygCBEEBazYCBAsgA0IAEOMCDAYLIAINAUEIIQIMAgsgAkEKIAIbIgIgBEGBmwFqLQAASw0AIAMpA3BCAFkEQCADIAMoAgRBAWs2AgQLIANCABDjAkG47AJBHDYCAAwECyACQQpHDQAgBEEwayIFQQlNBEBBACECA0AgAkEKbCECAn8gAygCBCIGIAMoAmhHBEAgAyAGQQFqNgIEIAYtAAAMAQsgAxDkAgshBCACIAVqIgJBmbPmzAFJIARBMGsiBUEJTXENAAsgAq0hEgsCQCAFQQlLDQAgEkIKfiEUIAWtIRMDQAJ/IAMoAgQiAiADKAJoRwRAIAMgAkEBajYCBCACLQAADAELIAMQ5AILIQQgEyAUfCESIARBMGsiBUEJSw0BIBJCmrPmzJmz5swZWg0BIBJCCn4iFCAFrSITQn+FWA0AC0EKIQIMAgtBCiECIAVBCU0NAQwCCyACIAJBAWtxBEAgBEGBmwFqLQAAIgYgAkkEQANAIAIgBWwhBQJ/IAMoAgQiBCADKAJoRwRAIAMgBEEBajYCBCAELQAADAELIAMQ5AILIQQgBSAGaiIFQcfj8ThJIARBgZsBai0AACIGIAJJcQ0ACyAFrSESCyACIAZNDQEgAq0hFgNAIBIgFn4iFCAGrUL/AYMiE0J/hVYNAgJ/IAMoAgQiBiADKAJoRwRAIAMgBkEBajYCBCAGLQAADAELIAMQ5AILIQQgEyAUfCESIAIgBEGBmwFqLQAAIgZNDQIgESAWQgAgEkIAEPACIBEpAwhQDQALDAELIAJBF2xBBXZBB3FBgZ0BaiwAACEMIARBgZsBai0AACIFIAJJBEADQCAGIAx0IQYCfyADKAIEIgQgAygCaEcEQCADIARBAWo2AgQgBC0AAAwBCyADEOQCCyEEIAUgBnIiBkGAgIDAAEkgBEGBmwFqLQAAIgUgAklxDQALIAatIRILIAIgBU0NAEJ/IAytIheIIhYgElQNAANAIBIgF4YhFCAFrUL/AYMhEwJ/IAMoAgQiBiADKAJoRwRAIAMgBkEBajYCBCAGLQAADAELIAMQ5AILIQQgEyAUhCESIAIgBEGBmwFqLQAAIgVNDQEgEiAWWA0ACwsgAiAEQYGbAWotAABNDQADQCACAn8gAygCBCIGIAMoAmhHBEAgAyAGQQFqNgIEIAYtAAAMAQsgAxDkAgtBgZsBai0AAEsNAAtBuOwCQcQANgIAQQAhDkJ/IRILIAMpA3BCAFkEQCADIAMoAgRBAWs2AgQLAkAgEkJ/Ug0ACyASIA6sIhOFIBN9IRILIBFBEGokACADKQN4QgAgAygCBCADKAIsa6x9UQ0HAkAgC0HwAEcNACAJRQ0AIAkgEj4CAAwDCyAJIBAgEhD3AgwCCyAJRQ0BIAcpAxAhFCAHKQMIIRMCQAJAAkAgEA4DAAECBAsgCSATIBQQ9QI4AgAMAwsgCSATIBQQ+QE5AwAMAgsgCSATNwMAIAkgFDcDCAwBC0EfIARBAWogC0HjAEciDBshBQJAIBBBAUYEQCAJIQIgCARAIAVBAnQQ8gEiAkUNBwsgB0IANwKoAkEAIQQDQCACIQACQANAAn8gAygCBCICIAMoAmhHBEAgAyACQQFqNgIEIAItAAAMAQsgAxDkAgsiAiAHai0AIUUNASAHIAI6ABsgB0EcaiAHQRtqQQEgB0GoAmoQ9gIiAkF+Rg0AQQAhCiACQX9GDQsgAARAIAAgBEECdGogBygCHDYCACAEQQFqIQQLIAhFDQAgBCAFRw0AC0EBIQYgACAFQQF0QQFyIgVBAnQQ9AEiAg0BDAsLC0EAIQogACEFIAdBqAJqBH8gBygCqAIFQQALDQgMAQsgCARAQQAhBCAFEPIBIgJFDQYDQCACIQADQAJ/IAMoAgQiAiADKAJoRwRAIAMgAkEBajYCBCACLQAADAELIAMQ5AILIgIgB2otACFFBEBBACEFIAAhCgwECyAAIARqIAI6AAAgBEEBaiIEIAVHDQALQQEhBiAAIAVBAXRBAXIiBRD0ASICDQALIAAhCkEAIQAMCQtBACEEIAkEQANAAn8gAygCBCIAIAMoAmhHBEAgAyAAQQFqNgIEIAAtAAAMAQsgAxDkAgsiACAHai0AIQRAIAQgCWogADoAACAEQQFqIQQMAQVBACEFIAkiACEKDAMLAAsACwNAAn8gAygCBCIAIAMoAmhHBEAgAyAAQQFqNgIEIAAtAAAMAQsgAxDkAgsgB2otACENAAtBACEAQQAhCkEAIQULIAMoAgQhAiADKQNwQgBZBEAgAyACQQFrIgI2AgQLIAMpA3ggAiADKAIsa6x8IhNQDQIgDCATIBRRckUNAiAIBEAgCSAANgIACwJAIAtB4wBGDQAgBQRAIAUgBEECdGpBADYCAAsgCkUEQEEAIQoMAQsgBCAKakEAOgAACyAFIQALIAMoAgQgAygCLGusIAMpA3ggFXx8IRUgDSAJQQBHaiENCyABQQFqIQQgAS0AASIBDQEMCAsLIAUhAAwBC0EBIQZBACEKQQAhAAwCCyAIIQYMAwsgCCEGCyANDQELQX8hDQsgBkUNACAKEPMBIAAQ8wELIAdBsAJqJAAgA0GQAWokACANC1UBAn8gASAAKAJUIgEgAUEAIAJBgAJqIgMQ1gEiBCABayADIAQbIgMgAiACIANLGyICENIBGiAAIAEgA2oiAzYCVCAAIAM2AgggACABIAJqNgIEIAILTQECfyABLQAAIQICQCAALQAAIgNFDQAgAiADRw0AA0AgAS0AASECIAAtAAEiA0UNASABQQFqIQEgAEEBaiEAIAIgA0YNAAsLIAMgAmsLKAAgAEEYdCAAQYD+A3FBCHRyIABBCHZBgP4DcSAAQRh2cnIgACABGwtmAQN/IAJFBEBBAA8LAkAgAC0AACIDRQ0AA0ACQCABLQAAIgVFDQAgAkEBayICRQ0AIAMgBUcNACABQQFqIQEgAC0AASEDIABBAWohACADDQEMAgsLIAMhBAsgBEH/AXEgAS0AAGsLgAEBBH8gACAAQT0Q/gEiAUYEQEEADwsCQCAAIAEgAGsiBGotAAANAEHw8QIoAgAiAUUNACABKAIAIgJFDQADQAJAIAAgAiAEEPwCRQRAIAEoAgAgBGoiAi0AAEE9Rg0BCyABKAIEIQIgAUEEaiEBIAINAQwCCwsgAkEBaiEDCyADC+gCAQN/AkAgAS0AAA0AQZguEP0CIgEEQCABLQAADQELIABBDGxB0J8BahD9AiIBBEAgAS0AAA0BC0GtLhD9AiIBBEAgAS0AAA0BC0HPNyEBCwJAA0ACQCABIAJqLQAAIgRFDQAgBEEvRg0AQRchBCACQQFqIgJBF0cNAQwCCwsgAiEEC0HPNyEDAkACQAJAAkACQCABLQAAIgJBLkYNACABIARqLQAADQAgASEDIAJBwwBHDQELIAMtAAFFDQELIANBzzcQ+gJFDQAgA0H+LBD6Ag0BCyAARQRAQfSeASECIAMtAAFBLkYNAgtBAA8LQfjxAigCACICBEADQCADIAJBCGoQ+gJFDQIgAigCICICDQALC0EkEPIBIgIEQCACQfSeASkCADcCACACQQhqIgEgAyAEENIBGiABIARqQQA6AAAgAkH48QIoAgA2AiBB+PECIAI2AgALIAJB9J4BIAAgAnIbIQILIAILogEBA38jAEGgAWsiBCQAIAQgACAEQZ4BaiABGyIFNgKUAUF/IQAgBCABQQFrIgZBACABIAZPGzYCmAEgBEEAQZABENQBIgRBfzYCTCAEQcMBNgIkIARBfzYCUCAEIARBnwFqNgIsIAQgBEGUAWo2AlQCQCABQQBIBEBBuOwCQT02AgAMAQsgBUEAOgAAIAQgAiADEO0BIQALIARBoAFqJAAgAAurAQEEfyAAKAJUIgMoAgQiBSAAKAIUIAAoAhwiBmsiBCAEIAVLGyIEBEAgAygCACAGIAQQ0gEaIAMgAygCACAEajYCACADIAMoAgQgBGsiBTYCBAsgAygCACEEIAUgAiACIAVLGyIFBEAgBCABIAUQ0gEaIAMgAygCACAFaiIENgIAIAMgAygCBCAFazYCBAsgBEEAOgAAIAAgACgCLCIBNgIcIAAgATYCFCACCykBAX8jAEEQayICJAAgAiABNgIMIABBgyQgARD4AiEAIAJBEGokACAACyoBAX8jAEEQayIEJAAgBCADNgIMIAAgASACIAMQ/wIhACAEQRBqJAAgAAsvACAAQQBHIABBmJ8BR3EgAEGwnwFHcSAAQfzxAkdxIABBlPICR3EEQCAAEPMBCwvRAQEBfwJAAkAgACABc0EDcQRAIAEtAAAhAgwBCyABQQNxBEADQCAAIAEtAAAiAjoAACACRQ0DIABBAWohACABQQFqIgFBA3ENAAsLIAEoAgAiAkF/cyACQYGChAhrcUGAgYKEeHENAANAIAAgAjYCACABKAIEIQIgAEEEaiEAIAFBBGohASACQYGChAhrIAJBf3NxQYCBgoR4cUUNAAsLIAAgAjoAACACQf8BcUUNAANAIAAgAS0AASICOgABIABBAWohACABQQFqIQEgAg0ACwsLIwECfyAAIQEDQCABIgJBBGohASACKAIADQALIAIgAGtBAnULswgBBX8gASgCACEEAkACQAJAAkACQAJAAkACfwJAAkACQAJAIANFDQAgAygCACIGRQ0AIABFBEAgAiEDDAMLIANBADYCACACIQMMAQsCQEHU7QIoAgAoAgBFBEAgAEUNASACRQ0MIAIhBgNAIAQsAAAiAwRAIAAgA0H/vwNxNgIAIABBBGohACAEQQFqIQQgBkEBayIGDQEMDgsLIABBADYCACABQQA2AgAgAiAGaw8LIAIhAyAARQ0DDAULIAQQ4wEPC0EBIQUMAwtBAAwBC0EBCyEFA0AgBUUEQCAELQAAQQN2IgVBEGsgBkEadSAFanJBB0sNAwJ/IARBAWoiBSAGQYCAgBBxRQ0AGiAFLQAAQcABcUGAAUcEQCAEQQFrIQQMBwsgBEECaiIFIAZBgIAgcUUNABogBS0AAEHAAXFBgAFHBEAgBEEBayEEDAcLIARBA2oLIQQgA0EBayEDQQEhBQwBCwNAIAQtAAAhBgJAIARBA3ENACAGQQFrQf4ASw0AIAQoAgAiBkGBgoQIayAGckGAgYKEeHENAANAIANBBGshAyAEKAIEIQYgBEEEaiEEIAYgBkGBgoQIa3JBgIGChHhxRQ0ACwsgBkH/AXEiBUEBa0H+AE0EQCADQQFrIQMgBEEBaiEEDAELCyAFQcIBayIFQTJLDQMgBEEBaiEEIAVBAnRBkJ0BaigCACEGQQAhBQwACwALA0AgBUUEQCADRQ0HA0ACQAJAAkAgBC0AACIFQQFrIgdB/gBLBEAgBSEGDAELIARBA3ENASADQQVJDQECQANAIAQoAgAiBkGBgoQIayAGckGAgYKEeHENASAAIAZB/wFxNgIAIAAgBC0AATYCBCAAIAQtAAI2AgggACAELQADNgIMIABBEGohACAEQQRqIQQgA0EEayIDQQRLDQALIAQtAAAhBgsgBkH/AXEiBUEBayEHCyAHQf4ASw0BCyAAIAU2AgAgAEEEaiEAIARBAWohBCADQQFrIgMNAQwJCwsgBUHCAWsiBUEySw0DIARBAWohBCAFQQJ0QZCdAWooAgAhBkEBIQUMAQsgBC0AACIFQQN2IgdBEGsgByAGQRp1anJBB0sNAQJAAkACfyAEQQFqIgcgBUGAAWsgBkEGdHIiBUEATg0AGiAHLQAAQYABayIHQT9LDQEgBEECaiIIIAcgBUEGdHIiBUEATg0AGiAILQAAQYABayIHQT9LDQEgByAFQQZ0ciEFIARBA2oLIQQgACAFNgIAIANBAWshAyAAQQRqIQAMAQtBuOwCQRk2AgAgBEEBayEEDAULQQAhBQwACwALIARBAWshBCAGDQEgBC0AACEGCyAGQf8BcQ0AIAAEQCAAQQA2AgAgAUEANgIACyACIANrDwtBuOwCQRk2AgAgAEUNAQsgASAENgIAC0F/DwsgASAENgIAIAILnwQCB38EfiMAQRBrIggkAAJAAkACQCACQSRMBEAgAC0AACIFDQEgACEEDAILQbjsAkEcNgIAQgAhAwwCCyAAIQQCQANAIAXAIgVBIEYgBUEJa0EFSXJFDQEgBC0AASEFIARBAWohBCAFDQALDAELAkAgBC0AACIFQStrDgMAAQABC0F/QQAgBUEtRhshByAEQQFqIQQLAn8CQCACQRByQRBHDQAgBC0AAEEwRw0AQQEhCSAELQABQd8BcUHYAEYEQCAEQQJqIQRBEAwCCyAEQQFqIQQgAkEIIAIbDAELIAJBCiACGwsiCq0hDEEAIQIDQAJAQVAhBQJAIAQsAAAiBkEwa0H/AXFBCkkNAEGpfyEFIAZB4QBrQf8BcUEaSQ0AQUkhBSAGQcEAa0H/AXFBGUsNAQsgBSAGaiIGIApODQAgCCAMQgAgC0IAEPACQQEhBQJAIAgpAwhCAFINACALIAx+Ig0gBq0iDkJ/hVYNACANIA58IQtBASEJIAIhBQsgBEEBaiEEIAUhAgwBCwsgAQRAIAEgBCAAIAkbNgIACwJAAkAgAgRAQbjsAkHEADYCACAHQQAgA0IBgyIMUBshByADIQsMAQsgAyALVg0BIANCAYMhDAsCQCAMpw0AIAcNAEG47AJBxAA2AgAgA0IBfSEDDAILIAMgC1oNAEG47AJBxAA2AgAMAQsgCyAHrCIDhSADfSEDCyAIQRBqJAAgAwt/AgJ/An4jAEGgAWsiBCQAIAQgATYCPCAEIAE2AhQgBEF/NgIYIARBEGoiBUIAEOMCIAQgBSADQQEQ8wIgBCkDCCEGIAQpAwAhByACBEAgAiABIAQoAhQgBCgCiAFqIAQoAjxrajYCAAsgACAGNwMIIAAgBzcDACAEQaABaiQAC14BA38gASAEIANraiEFAkADQCADIARHBEBBfyEAIAEgAkYNAiABLAAAIgYgAywAACIHSA0CIAYgB0oEQEEBDwUgA0EBaiEDIAFBAWohAQwCCwALCyACIAVHIQALIAALCwAgACACIAMQiwMLHQEBfyMAQRBrIgMkACAAIAEgAhC3AiADQRBqJAALQAEBf0EAIQADfyABIAJGBH8gAAUgASwAACAAQQR0aiIAQYCAgIB/cSIDQRh2IANyIABzIQAgAUEBaiEBDAELCwtUAQJ/AkADQCADIARHBEBBfyEAIAEgAkYNAiABKAIAIgUgAygCACIGSA0CIAUgBkoEQEEBDwUgA0EEaiEDIAFBBGohAQwCCwALCyABIAJHIQALIAALGwAjAEEQayIBJAAgACACIAMQjwMgAUEQaiQAC4kCAQN/IwBBEGsiBCQAIAIgAWtBAnUiBUHv////A00EQAJAIAVBAkkEQCAAIAAtAAtBgAFxIAVyOgALIAAgAC0AC0H/AHE6AAsgACEDDAELIARBCGogACAFQQJPBH8gBUEEakF8cSIDIANBAWsiAyADQQJGGwVBAQtBAWoQlQUgBCgCDBogACAEKAIIIgM2AgAgACAAKAIIQYCAgIB4cSAEKAIMQf////8HcXI2AgggACAAKAIIQYCAgIB4cjYCCCAAIAU2AgQLA0AgASACRwRAIAMgASgCADYCACADQQRqIQMgAUEEaiEBDAELCyAEQQA2AgQgAyAEKAIENgIAIARBEGokAA8LEDMAC0ABAX9BACEAA38gASACRgR/IAAFIAEoAgAgAEEEdGoiAEGAgICAf3EiA0EYdiADciAAcyEAIAFBBGohAQwBCwsLsAQBAn8jAEEgayIGJAAgBiABNgIYAkACQAJAIAMoAgRBAXFFBEAgBkF/NgIAIAAgASACIAMgBCAGIAAoAgAoAhARCAAhAQJAAkAgBigCAA4CAwABCyAFQQE6AAAMBAsgBUEBOgAAIARBBDYCAAwDCyAGIAMoAhwiADYCACAAIAAoAgRBAWo2AgQGQCAGEJ8CIQcMAhkgBiQAIAYoAgAiACAAKAIEQQFrIgE2AgQgAUF/RgRAIAAgACgCACgCCBECAAsJAAsACyAFQQA6AAAMAQsgBigCACIAIAAoAgRBAWsiATYCBCABQX9GBEAgACAAKAIAKAIIEQIACyAGIAMoAhwiADYCACAAIAAoAgRBAWo2AgQGQCAGEJIDIQAZIAYkACAGKAIAIgAgACgCBEEBayIBNgIEIAFBf0YEQCAAIAAoAgAoAggRAgALCQALIAYoAgAiASABKAIEQQFrIgM2AgQgA0F/RgRAIAEgASgCACgCCBECAAsgBiAGNgIcBkAgBiAAIAAoAgAoAhgRAAAgBiAGQQxyIgE2AhwgASAAIAAoAgAoAhwRAAAZIAYkACAGKAIcIgMgBkcEQANAIANBDGsQxwUiAyAGRw0ACwsJAAsGQCAGQRhqIgMgAiAGIAMgByAEQQEQkwMhABkgBiQAA0AgA0EMaxDHBSIDIAZHDQALCQALIAUgACAGRjoAACAGKAIYIQEDQCADQQxrEMcFIgMgBkcNAAsLIAZBIGokACABCwsAIABBuPQCEJQDC9sFAQt/IwBBgAFrIggkACAIIAE2AnwgAyACa0EMbSEJIAhBxAE2AgQgCEEIakEAIAhBBGoQvgIhDiAIQRBqIQoGQAJAIAlB5QBPBEAgCRDyASIKRQRAEKcFAAsgDiAKEJUDCyAKIQcgAiEBA0AgASADRgRAA0AgACAIQfwAahChAiAJRXJBAUYEQCAAIAhB/ABqEKECBEAgBSAFKAIAQQJyNgIACwwECwJ/IAAoAgAiBygCDCIBIAcoAhBGBEAgByAHKAIAKAIkEQEADAELIAEtAAALwCENIAZFBEAgBCANIAQoAgAoAgwRAwAhDQsgD0EBaiEMQQAhECAKIQcgAiEBA0AgASADRgRAIAwhDyAQRQ0CIAAQogIaIAohByACIQEgCSALakECSQ0CA0AgASADRg0DAkAgBy0AAEECRw0AAn8gAS0AC0EHdgRAIAEoAgQMAQsgAS0AC0H/AHELIA9GDQAgB0EAOgAAIAtBAWshCwsgB0EBaiEHIAFBDGohAQwACwALAkAgBy0AAEEBRw0AAn8gAS0AC0EHdgRAIAEoAgAMAQsgAQsgD2otAAAhEQJAIA1B/wFxIAYEfyARBSAEIBHAIAQoAgAoAgwRAwALQf8BcUYEQEEBIRACfyABLQALQQd2BEAgASgCBAwBCyABLQALQf8AcQsgDEcNAiAHQQI6AAAgC0EBaiELDAELIAdBADoAAAsgCUEBayEJCyAHQQFqIQcgAUEMaiEBDAALAAsABSAHQQJBAQJ/IAEtAAtBB3YEQCABKAIEDAELIAEtAAtB/wBxC0UiDBs6AAAgB0EBaiEHIAFBDGohASALIAxqIQsgCSAMayEJDAELAAsACxkgCCQAIA4QlgMJAAsCQAJAA0AgAiADRg0BIAotAABBAkcEQCAKQQFqIQogAkEMaiECDAELCyACIQMMAQsgBSAFKAIAQQRyNgIACyAOEJYDIAhBgAFqJAAgAwspACAAKAIAIgAgARC0BCIBELsERQRAEMICAAsgACgCCCABQQJ0aigCAAs0AQF/IAAoAgAhAiAAIAE2AgAgAgRAIwAhAQZAIAIgAEEEaigCABECABkgASQAEPAFAAsLCwkAIABBABCVAwu2BQECfyMAQYACayIAJAAgACACNgL4ASAAIAE2AvwBIAMQmAMhBiAAQcQBaiADIABB9wFqEJkDBkAjAEEQayICJAAgAEG4AWoiAUIANwIAIAFBADYCCCACQRBqJAAgASABLQALQQd2BH8gASgCCEH/////B3FBAWsFQQoLELgCIAACfyABLQALQQd2BEAgASgCAAwBCyABCyICNgK0ASAAIABBEGo2AgwgAEEANgIIA0ACQCAAQfwBaiAAQfgBahChAg0AIAAoArQBAn8gAS0AC0EHdgRAIAEoAgQMAQsgAS0AC0H/AHELIAJqRgRAAn8gAS0AC0EHdgRAIAEoAgQMAQsgAS0AC0H/AHELIQMgAQJ/IAEtAAtBB3YEQCABKAIEDAELIAEtAAtB/wBxC0EBdBC4AiABIAEtAAtBB3YEfyABKAIIQf////8HcUEBawVBCgsQuAIgACADAn8gAS0AC0EHdgRAIAEoAgAMAQsgAQsiAmo2ArQBCwJ/IAAoAvwBIgMoAgwiByADKAIQRgRAIAMgAygCACgCJBEBAAwBCyAHLQAAC8AgBiACIABBtAFqIABBCGogACwA9wEgAEHEAWogAEEQaiAAQQxqQbC4ARCaAw0AIABB/AFqEKICGgwBCwsCQAJ/IAAtAM8BQQd2BEAgACgCyAEMAQsgAC0AzwFB/wBxC0UNACAAKAIMIgMgAEEQamtBnwFKDQAgACADQQRqNgIMIAMgACgCCDYCAAsgBSACIAAoArQBIAQgBhCbAzYCACAAQcQBaiAAQRBqIAAoAgwgBBCcAyAAQfwBaiAAQfgBahChAiECGSAAJAAgARDHBRogAEHEAWoQxwUaCQALIAIEQCAEIAQoAgBBAnI2AgALIAAoAvwBIQIgARDHBRogAEHEAWoQxwUaIABBgAJqJAAgAgsuAAJAIAAoAgRBygBxIgAEQCAAQcAARgRAQQgPCyAAQQhHDQFBEA8LQQAPC0EKC7QBAQJ/IwBBEGsiAyQAIANBDGoiBCABKAIcIgE2AgAgASABKAIEQQFqNgIEBkAgAiAEEJIDIgEgASgCACgCEBEBADoAACAAIAEgASgCACgCFBEAABkgAyQAIAMoAgwiACAAKAIEQQFrIgE2AgQgAUF/RgRAIAAgACgCACgCCBECAAsJAAsgAygCDCIAIAAoAgRBAWsiATYCBCABQX9GBEAgACAAKAIAKAIIEQIACyADQRBqJAALjAMBA38jAEEQayIKJAAgCiAAOgAPAkACQAJAIAMoAgAgAkcNAEErIQsgAEH/AXEiDCAJLQAYRwRAQS0hCyAJLQAZIAxHDQELIAMgAkEBajYCACACIAs6AAAMAQsCQAJ/IAYtAAtBB3YEQCAGKAIEDAELIAYtAAtB/wBxC0UNACAAIAVHDQBBACEAIAgoAgAiASAHa0GfAUoNAiAEKAIAIQAgCCABQQRqNgIAIAEgADYCAAwBC0F/IQAgCSAJQRpqIApBD2oQsAMgCWsiBUEXSg0BAkACQAJAIAFBCGsOAwACAAELIAEgBUoNAQwDCyABQRBHDQAgBUEWSA0AIAMoAgAiASACRg0CIAEgAmtBAkoNAiABQQFrLQAAQTBHDQJBACEAIARBADYCACADIAFBAWo2AgAgASAFQbC4AWotAAA6AAAMAgsgAyADKAIAIgBBAWo2AgAgACAFQbC4AWotAAA6AAAgBCAEKAIAQQFqNgIAQQAhAAwBC0EAIQAgBEEANgIACyAKQRBqJAAgAAvFAQICfwF+IwBBEGsiBCQAAn8CQAJAIAAgAUcEQEG47AIoAgAhBUG47AJBADYCACAAIARBDGogAxCuAxCkBSEGAkBBuOwCKAIAIgAEQCAEKAIMIAFHDQEgAEHEAEYNBAwDC0G47AIgBTYCACAEKAIMIAFGDQILCyACQQQ2AgBBAAwCCyAGQoCAgIB4Uw0AIAZC/////wdVDQAgBqcMAQsgAkEENgIAQf////8HIAZCAFUNABpBgICAgHgLIQAgBEEQaiQAIAAL8AEBAn8CfyAALQALQQd2BEAgACgCBAwBCyAALQALQf8AcQshBAJAIAIgAWtBBUgNACAERQ0AIAEgAhDpAyACQQRrIQQCfyAALQALQQd2BEAgACgCBAwBCyAALQALQf8AcQsCfyAALQALQQd2BEAgACgCAAwBCyAACyICaiEFAkADQAJAIAIsAAAhACABIARPDQACQCAAQQBMDQAgAEH/AE4NACABKAIAIAIsAABHDQMLIAFBBGohASACIAUgAmtBAUpqIQIMAQsLIABBAEwNASAAQf8ATg0BIAIsAAAgBCgCAEEBa0sNAQsgA0EENgIACwu2BQECfyMAQYACayIAJAAgACACNgL4ASAAIAE2AvwBIAMQmAMhBiAAQcQBaiADIABB9wFqEJkDBkAjAEEQayICJAAgAEG4AWoiAUIANwIAIAFBADYCCCACQRBqJAAgASABLQALQQd2BH8gASgCCEH/////B3FBAWsFQQoLELgCIAACfyABLQALQQd2BEAgASgCAAwBCyABCyICNgK0ASAAIABBEGo2AgwgAEEANgIIA0ACQCAAQfwBaiAAQfgBahChAg0AIAAoArQBAn8gAS0AC0EHdgRAIAEoAgQMAQsgAS0AC0H/AHELIAJqRgRAAn8gAS0AC0EHdgRAIAEoAgQMAQsgAS0AC0H/AHELIQMgAQJ/IAEtAAtBB3YEQCABKAIEDAELIAEtAAtB/wBxC0EBdBC4AiABIAEtAAtBB3YEfyABKAIIQf////8HcUEBawVBCgsQuAIgACADAn8gAS0AC0EHdgRAIAEoAgAMAQsgAQsiAmo2ArQBCwJ/IAAoAvwBIgMoAgwiByADKAIQRgRAIAMgAygCACgCJBEBAAwBCyAHLQAAC8AgBiACIABBtAFqIABBCGogACwA9wEgAEHEAWogAEEQaiAAQQxqQbC4ARCaAw0AIABB/AFqEKICGgwBCwsCQAJ/IAAtAM8BQQd2BEAgACgCyAEMAQsgAC0AzwFB/wBxC0UNACAAKAIMIgMgAEEQamtBnwFKDQAgACADQQRqNgIMIAMgACgCCDYCAAsgBSACIAAoArQBIAQgBhCeAzcDACAAQcQBaiAAQRBqIAAoAgwgBBCcAyAAQfwBaiAAQfgBahChAiECGSAAJAAgARDHBRogAEHEAWoQxwUaCQALIAIEQCAEIAQoAgBBAnI2AgALIAAoAvwBIQIgARDHBRogAEHEAWoQxwUaIABBgAJqJAAgAgu3AQIBfgJ/IwBBEGsiBSQAAkACQCAAIAFHBEBBuOwCKAIAIQZBuOwCQQA2AgAgACAFQQxqIAMQrgMQpAUhBAJAQbjsAigCACIABEAgBSgCDCABRw0BIABBxABGDQMMBAtBuOwCIAY2AgAgBSgCDCABRg0DCwsgAkEENgIAQgAhBAwBCyACQQQ2AgAgBEIAVQRAQv///////////wAhBAwBC0KAgICAgICAgIB/IQQLIAVBEGokACAEC7YFAQJ/IwBBgAJrIgAkACAAIAI2AvgBIAAgATYC/AEgAxCYAyEGIABBxAFqIAMgAEH3AWoQmQMGQCMAQRBrIgIkACAAQbgBaiIBQgA3AgAgAUEANgIIIAJBEGokACABIAEtAAtBB3YEfyABKAIIQf////8HcUEBawVBCgsQuAIgAAJ/IAEtAAtBB3YEQCABKAIADAELIAELIgI2ArQBIAAgAEEQajYCDCAAQQA2AggDQAJAIABB/AFqIABB+AFqEKECDQAgACgCtAECfyABLQALQQd2BEAgASgCBAwBCyABLQALQf8AcQsgAmpGBEACfyABLQALQQd2BEAgASgCBAwBCyABLQALQf8AcQshAyABAn8gAS0AC0EHdgRAIAEoAgQMAQsgAS0AC0H/AHELQQF0ELgCIAEgAS0AC0EHdgR/IAEoAghB/////wdxQQFrBUEKCxC4AiAAIAMCfyABLQALQQd2BEAgASgCAAwBCyABCyICajYCtAELAn8gACgC/AEiAygCDCIHIAMoAhBGBEAgAyADKAIAKAIkEQEADAELIActAAALwCAGIAIgAEG0AWogAEEIaiAALAD3ASAAQcQBaiAAQRBqIABBDGpBsLgBEJoDDQAgAEH8AWoQogIaDAELCwJAAn8gAC0AzwFBB3YEQCAAKALIAQwBCyAALQDPAUH/AHELRQ0AIAAoAgwiAyAAQRBqa0GfAUoNACAAIANBBGo2AgwgAyAAKAIINgIACyAFIAIgACgCtAEgBCAGEKADOwEAIABBxAFqIABBEGogACgCDCAEEJwDIABB/AFqIABB+AFqEKECIQIZIAAkACABEMcFGiAAQcQBahDHBRoJAAsgAgRAIAQgBCgCAEECcjYCAAsgACgC/AEhAiABEMcFGiAAQcQBahDHBRogAEGAAmokACACC90BAgN/AX4jAEEQayIEJAACfwJAAkACQCAAIAFHBEACQAJAIAAtAAAiBUEtRw0AIABBAWoiACABRw0ADAELQbjsAigCACEGQbjsAkEANgIAIAAgBEEMaiADEK4DEKUFIQcCQEG47AIoAgAiAARAIAQoAgwgAUcNASAAQcQARg0FDAQLQbjsAiAGNgIAIAQoAgwgAUYNAwsLCyACQQQ2AgBBAAwDCyAHQv//A1gNAQsgAkEENgIAQf//AwwBC0EAIAenIgBrIAAgBUEtRhsLIQAgBEEQaiQAIABB//8DcQu2BQECfyMAQYACayIAJAAgACACNgL4ASAAIAE2AvwBIAMQmAMhBiAAQcQBaiADIABB9wFqEJkDBkAjAEEQayICJAAgAEG4AWoiAUIANwIAIAFBADYCCCACQRBqJAAgASABLQALQQd2BH8gASgCCEH/////B3FBAWsFQQoLELgCIAACfyABLQALQQd2BEAgASgCAAwBCyABCyICNgK0ASAAIABBEGo2AgwgAEEANgIIA0ACQCAAQfwBaiAAQfgBahChAg0AIAAoArQBAn8gAS0AC0EHdgRAIAEoAgQMAQsg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AEECdEH88QJqIABB+tMAEP4CNgIAIABBAWoiAEEGRw0AC0Gs8gJBAToAAEGU8gJB/PECKAIANgIAC0H88QIhACABQQhqQfzxAkEYENcBRQ0BQZTyAiEAIAFBCGpBlPICQRgQ1wFFDQFBGBDyASIARQ0AIAAgASkCCDcCACAAIAEpAhg3AhAgACABKQIQNwIIDAELQQAhAAsgAUEgaiQAQdjzAkEBOgAAQdTzAiAANgIAIAALbAEBfyMAQRBrIgMkACADIAE2AgwgAyACNgIIIANBBGogA0EMahCxAyEBIABB3hogAygCCBD4AiECIAEoAgAiAARAQdTtAigCABogAARAQdTtAkHc7AIgACAAQX9GGzYCAAsLIANBEGokACACCzEAIAItAAAhAgNAAkAgACABRwR/IAAtAAAgAkcNASAABSABCw8LIABBAWohAAwACwALPQEBf0HU7QIoAgAhAiABKAIAIgEEQEHU7QJB3OwCIAEgAUF/Rhs2AgALIABBfyACIAJB3OwCRhs2AgAgAAuwBAECfyMAQSBrIgYkACAGIAE2AhgCQAJAAkAgAygCBEEBcUUEQCAGQX82AgAgACABIAIgAyAEIAYgACgCACgCEBEIACEBAkACQCAGKAIADgIDAAELIAVBAToAAAwECyAFQQE6AAAgBEEENgIADAMLIAYgAygCHCIANgIAIAAgACgCBEEBajYCBAZAIAYQsAIhBwwCGSAGJAAgBigCACIAIAAoAgRBAWsiATYCBCABQX9GBEAgACAAKAIAKAIIEQIACwkACwALIAVBADoAAAwBCyAGKAIAIgAgACgCBEEBayIBNgIEIAFBf0YEQCAAIAAoAgAoAggRAgALIAYgAygCHCIANgIAIAAgACgCBEEBajYCBAZAIAYQswMhABkgBiQAIAYoAgAiACAAKAIEQQFrIgE2AgQgAUF/RgRAIAAgACgCACgCCBECAAsJAAsgBigCACIBIAEoAgRBAWsiAzYCBCADQX9GBEAgASABKAIAKAIIEQIACyAGIAY2AhwGQCAGIAAgACgCACgCGBEAACAGIAZBDHIiATYCHCABIAAgACgCACgCHBEAABkgBiQAIAYoAhwiAyAGRwRAA0AgA0EMaxDQBSIDIAZHDQALCwkACwZAIAZBGGoiAyACIAYgAyAHIARBARC0AyEAGSAGJAADQCAD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AhkgACQAIAEQxwUaIABBxAFqEMcFGgkACyACBEAgBCAEKAIAQQJyNgIACyAAKALMAiECIAEQxwUaIABBxAFqEMcFGiAAQdACaiQAIAILwAUBA38jAEHQAmsiACQAIAAgAjYCyAIgACABNgLMAiADEJgDIQYgAyAAQdABahC2AyEHIABBxAFqIAMgAEHEAmoQtwMGQCMAQRBrIgIkACAAQbgBaiIBQgA3AgAgAUEANgIIIAJBEGokACABIAEtAAtBB3YEfyABKAIIQf////8HcUEBawVBCgsQuAIgAAJ/IAEtAAtBB3YEQCABKAIADAELIAELIgI2ArQBIAAgAEEQajYCDCAAQQA2AggDQAJAIABBzAJqIABByAJqELECDQAgACgCtAECfyABLQALQQd2BEAgASgCBAwBCyABLQALQf8AcQsgAmpGBEACfyABLQALQQd2BEAgASgCBAwBCyABLQALQf8AcQshAyABAn8gAS0AC0EHdgRAIAEoAgQMAQsgAS0AC0H/AHELQQF0ELgCIAEgAS0AC0EHdgR/IAEoAghB/////wdxQQFrBUEKCxC4AiAAIAMCfyABLQALQQd2BEAgASgCAAwBCyABCyICajYCtAELAn8gACgCzAIiAygCDCIIIAMoAhBGBEAgAyADKAIAKAIkEQEADAELIAgoAgALIAYgAiAAQbQBaiAAQQhqIAAoAsQCIABBxAFqIABBEGogAEEMaiAHELgDDQAgAEHMAmoQsgIaDAELCwJAAn8gAC0AzwFBB3YEQCAAKALIAQwBCyAALQDPAUH/AHELRQ0AIAAoAgwiAyAAQRBqa0GfAUoNACAAIANBBGo2AgwgAyAAKAIINgIACyAFIAIgACgCtAEgBCAGEKIDNgIAIABBxAFqIABBEGogACgCDCAEEJwDIABBzAJqIABByAJqELECIQIZIAAkACABEMcFGiAAQcQBahDHBRoJAAsgAgRAIAQgBCgCAEECcjYCAAsgACgCzAIhAiABEMcFGiAAQcQBahDHBRogAEHQAmokACACC8AFAQN/IwBB0AJrIgAkACAAIAI2AsgCIAAgATYCzAIgAxCYAyEGIAMgAEHQAWoQtgMhByAAQcQBaiADIABBxAJqELcDBkAjAEEQayICJAAgAEG4AWoiAUIANwIAIAFBADYCCCACQRBqJAAgASABLQALQQd2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AgALIAAoAvwCIQIgARDHBRogAEHcAWoQxwUaIABBgANqJAAgAguyBgECfyMAQcACayIAJAAgACACNgK4AiAAIAE2ArwCIwBBEGsiASQAIABBxAFqIgZCADcCACAGQQA2AgggAUEQaiQABkAgAEEQaiICIAMoAhwiATYCACABIAEoAgRBAWo2AgQGQCACELACIgFBsLgBQcq4ASAAQdABaiABKAIAKAIwEQYAGhkgACQABkAgACgCECIBIAEoAgRBAWsiAjYCBCACQX9GBEAgASABKAIAKAIIEQIACxgCCQALBkAgACgCECIBIAEoAgRBAWsiAjYCBCACQX9GBEAgASABKAIAKAIIEQIACxgBBkAjAEEQayIBJAAgAEG4AWoiAkIANwIAIAJBADYCCCABQRBqJAAgAiACLQALQQd2BH8gAigCCEH/////B3FBAWsFQQoLELgCIAACfyACLQALQQd2BEAgAigCAAwBCyACCyIBNgK0ASAAIABBEGo2AgwgAEEANgIIA0ACQCAAQbwCaiAAQbgCahCxAg0AIAAoArQBAn8gAi0AC0EHdgRAIAIoAgQMAQsgAi0AC0H/AHELIAFqRgRAAn8gAi0AC0EHdgRAIAIoAgQMAQsgAi0AC0H/AHELIQMgAgJ/IAItAAtBB3YEQCACKAIEDAELIAItAAtB/wBxC0EBdBC4AiACIAItAAtBB3YEfyACKAIIQf////8HcUEBawVBCgsQuAIgACADAn8gAi0AC0EHdgRAIAIoAgAMAQsgAgsiAWo2ArQBCwJ/IAAoArwCIgMoAgwiByADKAIQRgRAIAMgAygCACgCJBEBAAwBCyAHKAIAC0EQIAEgAEG0AWogAEEIakEAIAYgAEEQaiAAQQxqIABB0AFqELgDDQAgAEG8AmoQsgIaDAELCyACIAAoArQBIAFrELgCAn8gAi0AC0EHdgRAIAIoAgAMAQsgAgshARCuAyEDIAAgBTYCACABIAMgABCvA0EBRwRAIARBBDYCAAsgAEG8AmogAEG4AmoQsQIhARkgACQAIAIQxwUaCQALGSAAJAAgBhDHBRoJAAsgAQRAIAQgBCgCAEECcjYCAAsgACgCvAIhASACEMcFGiAGEMcFGiAAQcACaiQAIAELMQAgAigCACECA0ACQCAAIAFHBH8gACgCACACRw0BIAAFIAELDwsgAEEE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IBEgC0HkAGogC0HgAGoQkAQgCygCZCEKCyALIApBBGo2AmQgCiABNgIACwJAIAsoAhRBAEwNAAJAIAAgC0GMBGoQsQJFBEACfyAAKAIAIgEoAgwiBCABKAIQRgRAIAEgASgCACgCJBEBAAwBCyAEKAIACyEBIAEgCygCWEYNAQsMAwsgABCyAhoDQCALKAIUQQBMDQECQCAAIAtBjARqELECRQRAIAdBwAACfyAAKAIAIgEoAgwiBCABKAIQRgRAIAEgASgCACgCJBEBAAwBCyAEKAIACyAHKAIAKAIMEQQADQELDAQLIAkoAgAgCygCiARGBEAgCCAJIAtBiARqEJAECwJ/IAAoAgAiASgCDCIEIAEoAhBGBEAgASABKAIAKAIkEQEADAELIAQoAgALIQEgCSAJKAIAIgRBBGo2AgAgBCABNgIAIAsgCygCFEEBazYCFCAAELICGgwACwALIAIhCiAIKAIAIAkoAgBHDQQMAQsCQCACRQ0AQQEhCgNAAn8gAi0AC0EHdgRAIAIoAgQMAQsgAi0AC0H/AHELIApNDQECQCAAIAtBjARqELECRQRAAn8gACgCACIBKAIMIgMgASgCEEYEQCABIAEoAgAoAiQRAQAMAQsgAygCAAshASABAn8gAi0AC0EHdgRAIAIoAgAMAQsgAgsgCkECdGooAgBGDQELDAMLIApBAWohCiAAELICGgwACwALQQEhACARKAIAIAsoAmRGDQEgC0EANgIMIA8gESgCACALKAJkIAtBDGoQnAMgCygCDEUNAQsgBSAFKAIAQQRyNgIAQQAhAAsgEBDQBRogDBDQBRogDRDQBRogDhDQBRogDxDHBRogERCWAwwDCyACIQoLIANBAWohAwwACwALGSALJAAgEBDQBRogDBDQBRogDRDQBRogDhDQBRogDxDHBRogERCWAwkACwsgC0GQBGokACAACx8BAX8gASgCABCzAiECIAAgASgCADYCBCAAIAI2AgALkAQBA38jAEHAA2siACQAIAAgAjYCuAMgACABNgK8AyAAQcUBNgIUIABBGGogAEEgaiAAQRRqIgkQvgIhBwZAIABBEGoiCCAEKAIcIgE2AgAgASABKAIEQQFqNgIEBkAgCBCwAiEBIABBADoADyAAQbwDaiACIAMgCCAEKAIEIAUgAEEPaiABIAcgCSAAQbADahCZ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AQsgCC0AC0H/AHELIAkgD2tBAXRqampBAWoMAQsgACgCnAECfyAILQALQQd2BEAgCCgCBAwBCyAILQALQf8AcQsCfyAHLQALQQd2BEAgBygCBAwBCyAHLQALQf8AcQtqakECagsiD0HlAEkNACAMIA8Q8gEQlQMgDCgCACICDQAQpwUMBQsgAiAAQSRqIABBIGogAygCBCALIAkgC2ogECARIABByAFqIAAsAMcBIAAsAMYBIA4gByAIIAAoApwBEKMEIAEgAiAAKAIkIAAoAiAgAyAEEMwDIQIZIAAkACAMEJYDCQALGSAAJAAgCBDHBRogBxDHBRogDhDHBRoJAAsZIAAkAAZAIAAoAswBIgEgASgCBEEBayICNgIEIAJBf0YEQCABIAEoAgAoAggRAgALGAMJAAsZIAAkACAKEJYDIA0QlgMJAAsgDBCWAyAIEMcFGiAHEMcFGiAOEMcFGiAAKALMASIBIAEoAgRBAWsiAzYCBCADQX9GBEAgASABKAIAKAIIEQIACyAKEJYDIA0QlgMgAEHAA2okACACDwsAC+8DAQF/IwBBEGsiCiQAIAkCfyAABEAgAhCUBCEAAkAgAQRAIApBBGoiASAAIAAoAgAoAiwRAAAgAyAKKAIENgAAIAEgACAAKAIAKAIgEQAADAELIApBBGoiASAAIAAoAgAoAigRAAAgAyAKKAIENgAAIAEgACAAKAIAKAIcEQAACyAIIAEQtgIgARDHBRogBCAAIAAoAgAoAgwRAQA6AAAgBSAAIAAoAgAoAhARAQA6AAAgCkEEaiIBIAAgACgCACgCFBEAACAGIAEQtgIgARDHBRogASAAIAAoAgAoAhgRAAAgByABELYCIAEQxwUaIAAgACgCACgCJBEBAAwBCyACEJUEIQACQCABBEAgCkEEaiIBIAAgACgCACgCLBEAACADIAooAgQ2AAAgASAAIAAoAgAoAiARAAAMAQsgCkEEaiIBIAAgACgCACgCKBEAACADIAooAgQ2AAAgASAAIAAoAgAoAhwRAAALIAggARC2AiABEMcFGiAEIAAgACgCACgCDBEBADoAACAFIAAgACgCACgCEBEBADoAACAKQQRqIgEgACAAKAIAKAIUEQAAIAYgARC2AiABEMcFGiABIAAgACgCACgCGBEAACAHIAEQtgIgARDHBRogACAAKAIAKAIkEQEACzYCACAKQRBqJAAL4QcBCn8jAEEQayITJAAgAiAANgIAIANBgARxIRYDQCAUQQRGBEACfyANLQALQQd2BEAgDSgCBAwBCyANLQALQf8AcQtBAUsEQCATIA0QxQM2AgwgAiATQQxqQQEQlgQgDRDGAyACKAIAEKQENgIACyADQbABcSIDQRBHBEAgASADQSBGBH8gAigCAAUgAAs2AgALIBNBEGokAA8LAkACQAJAAkACQAJAIAggFGosAAAOBQABAwIEBQsgASACKAIANgIADAQLIAEgAigCADYCACAGQSAgBigCACgCHBEDACEPIAIgAigCACIQQQFqNgIAIBAgDzoAAAwDCwJ/IA0tAAtBB3YEQCANKAIEDAELIA0tAAtB/wBxC0UNAgJ/IA0tAAtBB3YEQCANKAIADAELIA0LLQAAIQ8gAiACKAIAIhBBAWo2AgAgECAPOgAADAILAn8gDC0AC0EHdgRAIAwoAgQMAQsgDC0AC0H/AHELRSEPIBZFDQEgDw0BIAIgDBDFAyAMEMYDIAIoAgAQpAQ2AgAMAQsgAigCACEXIAQgB2oiBCERA0ACQCAFIBFNDQAgESwAACIPQQBOBH8gBigCCCAPQf8BcUECdGooAgBBwABxQQBHBUEAC0UNACARQQFqIREMAQsLIA4iD0EASgRAA0ACQCAEIBFPDQAgD0UNACAPQQFrIQ8gEUEBayIRLQAAIRAgAiACKAIAIhJBAWo2AgAgEiAQOgAADAELCyAPBH8gBkEwIAYoAgAoAhwRAwAFQQALIRIDQCACIAIoAgAiEEEBajYCACAPQQBKBEAgECASOgAAIA9BAWshDwwBCwsgECAJOgAACwJAIAQgEUYEQCAGQTAgBigCACgCHBEDACEPIAIgAigCACIQQQFqNgIAIBAgDzoAAAwBCwJ/IAstAAtBB3YEQCALKAIEDAELIAstAAtB/wBxCwR/An8gCy0AC0EHdgRAIAsoAgAMAQsgCwssAAAFQX8LIRJBACEPQQAhEANAIAQgEUYNAQJAIA8gEkcEQCAPIRUMAQsgAiACKAIAIhJBAWo2AgAgEiAKOgAAQQAhFQJ/IAstAAtBB3YEQCALKAIEDAELIAstAAtB/wBxCyAQQQFqIhBNBEAgDyESDAELAn8gCy0AC0EHdgRAIAsoAgAMAQsgCwsgEGotAABB/wBGBEBBfyESDAELAn8gCy0AC0EHdgRAIAsoAgAMAQsgCwsgEGosAAAhEgsgEUEBayIRLQAAIQ8gAiACKAIAIhhBAWo2AgAgGCAPOgAAIBVBAWohDwwACwALIBcgAigCABDoAwsgFEEBaiEUDAALAAvMAQEDfyMAQRBrIgUkACMAQSBrIgMkACADQRhqIAAgARCbBSADQRBqIAMoAhggAygCHCACENICIAMoAhAhBCMAQRBrIgEkACABIAA2AgwgAUEMaiIAIAQgACgCACEEIwBBEGsiACQAIAAgBDYCDCAAKAIMIQQgAEEQaiQAIARrEJYEIQAgAUEQaiQAIAMgADYCDCADIAIgAygCFCACa2o2AgggBSADKAIMNgIIIAUgAygCCDYCDCADQSBqJAAgBSgCDCEAIAVBEGokACAAC/8GAQh/IwBBsAFrIgYkACAGQawBaiIIIAMoAhwiADYCACAAIAAoAgRBAWo2AgQGQCAIEJ8CIQoCfyAFLQALQQd2BEAgBSgCBAwBCyAFLQALQf8AcQsEfwJ/IAUtAAtBB3YEQCAFKAIADAELIAULLQAAIApBLSAKKAIAKAIcEQMAQf8BcUYFQQALIQwGQCMAQRBrIgAkACAGQZgBaiIIQgA3AgAgCEEANgIIIABBEGokACMAQRBrIgAkACAGQYwBaiIHQgA3AgAgB0EANgIIIABBEGokACACIAwgBkGsAWogBkGoAWogBkGnAWogBkGmAWogCCAHIwBBEGsiAiQAIAZBgAFqIgBCADcCACAAQQA2AgggAkEQaiQAIAAgBkH8AGoQogQgBkHEATYCECAGQQhqQQAgBkEQaiICEL4CIQkGQAJAAn8CfyAFLQALQQd2BEAgBSgCBAwBCyAFLQALQf8AcQsgBigCfEoEQAJ/IAUtAAtBB3YEQCAFKAIEDAELIAUtAAtB/wBxCyELIAYoAnwiDQJ/IActAAtBB3YEQCAHKAIEDAELIActAAtB/wBxCwJ/IAAtAAtBB3YEQCAAKAIEDAELIAAtAAtB/wBxCyALIA1rQQF0ampqQQFqDAELIAYoAnwCfyAALQALQQd2BEAgACgCBAwBCyAALQALQf8AcQsCfyAHLQALQQd2BEAgBygCBAwBCyAHLQALQf8AcQtqakECagsiC0HlAEkNACAJIAsQ8gEQlQMgCSgCACICDQAQpwUACyACIAZBBGogBiADKAIEAn8gBS0AC0EHdgRAIAUoAgAMAQsgBQsCfyAFLQALQQd2BEAgBSgCAAwBCyAFCwJ/IAUtAAtBB3YEQCAFKAIEDAELIAUtAAtB/wBxC2ogCiAMIAZBqAFqIAYsAKcBIAYsAKYBIAggByAAIAYoAnwQowQgASACIAYoAgQgBigCACADIAQQzAMhARkgBiQAIAkQlgMJAAsZIAYkACAAEMcFGiAHEMcFGiAIEMcFGgkACxkgBiQAIAYoAqwBIgAgACgCBEEBayIBNgIEIAFBf0YEQCAAIAAoAgAoAggRAgALCQALIAkQlgMgABDHBRogBxDHBRogCBDHBRogBigCrAEiACAAKAIEQQFrIgI2AgQgAkF/RgRAIAAgACgCACgCCBECAAsgBkGwAWokACABC+cHAQt/IwBBoAhrIgAkACAAIAU3AxAgACAGNwMYIAAgAEGwB2oiBzYCrAcgB0HkAEH9IyAAQRBqEIIDIQkgAEHEATYCMCAAQYgEakEAIABBMGoiBxC+AiENIABBxAE2AjAgAEGABGpBACAHEL4CIQogAEGQBGohCwJABkAgCUHkAE8EQBCuAyEHIAAgBTcDACAAIAY3AwggAEGsB2ogB0H9IyAAENMDIglBf0YEQBCnBQwDCyANIAAoAqwHEJUDIAogCUECdBDyARCVAyAKKAIARQRAEKcFDAMLIAooAgAhCwsgAEH8A2oiCCADKAIcIgc2AgAgByAHKAIEQQFqNgIEBkAgCBCwAiIQIgcgACgCrAciCCAIIAlqIAsgBygCACgCMBEGABoGQCACIAlBAEwEf0EABSAAKAKsBy0AAEEtRgsiESAAQfwDaiAAQfgDaiAAQfQDaiAAQfADaiMAQRBrIgckACAAQeQDaiICQgA3AgAgAkEANgIIIAdBEGokACACIg4jAEEQayIHJAAgAEHYA2oiAkIANwIAIAJBADYCCCAHQRBqJAAgAiIHIwBBEGsiCCQAIABBzANqIgJCADcCACACQQA2AgggCEEQaiQAIAIiCCAAQcgDahCnBCAAQcQBNgIwIABBKGpBACAAQTBqIgIQvgIhDAZAAkACfyAAKALIAyIPIAlIBEAgACgCyAMCfyAHLQALQQd2BEAgBygCBAwBCyAHLQALQf8AcQsCfyAILQALQQd2BEAgCCgCBAwBCyAILQALQf8AcQsgCSAPa0EBdGpqakEBagwBCyAAKALIAwJ/IAgtAAtBB3YEQCAIKAIEDAELIAgtAAtB/wBxCwJ/IActAAtBB3YEQCAHKAIEDAELIActAAtB/wBxC2pqQQJqCyIPQeUASQ0AIAwgD0ECdBDyARCVAyAMKAIAIgINABCnBQwFCyACIABBJGogAEEgaiADKAIEIAsgCyAJQQJ0aiAQIBEgAEH4A2ogACgC9AMgACgC8AMgDiAHIAggACgCyAMQqAQgASACIAAoAiQgACgCICADIAQQ3QMhAhkgACQAIAwQlgMJAAsZIAAkACAIENAFGiAHENAFGiAOEMcFGgkACxkgACQABkAgACgC/AMiASABKAIEQQFrIgI2AgQgAkF/RgRAIAEgASgCACgCCBECAAsYAwkACxkgACQAIAoQlgMgDRCWAwkACyAMEJYDIAgQ0AUaIAcQ0AUaIA4QxwUaIAAoAvwDIgEgASgCBEEBayIDNgIEIANBf0YEQCABIAEoAgAoAggRAgALIAoQlgMgDRCWAyAAQaAIaiQAIAIPCwAL7wMBAX8jAEEQayIKJAAgCQJ/IAAEQCACEJ0EIQACQCABBEAgCkEEaiIBIAAgACgCACgCLBEAACADIAooAgQ2AAAgASAAIAAoAgAoAiARAAAMAQsgCkEEaiIBIAAgACgCACgCKBEAACADIAooAgQ2AAAgASAAIAAoAgAoAhwRAAALIAggARCeBCABENAFGiAEIAAgACgCACgCDBEBADYCACAFIAAgACgCACgCEBEBADYCACAKQQRqIgEgACAAKAIAKAIUEQAAIAYgARC2AiABEMcFGiABIAAgACgCACgCGBEAACAHIAEQngQgARDQBRogACAAKAIAKAIkEQEADAELIAIQnwQhAAJAIAEEQCAKQQRqIgEgACAAKAIAKAIsEQAAIAMgCigCBDYAACABIAAgACgCACgCIBEAAAwBCyAKQQRqIgEgACAAKAIAKAIoEQAAIAMgCigCBDYAACABIAAgACgCACgCHBEAAAsgCCABEJ4EIAEQ0AUaIAQgACAAKAIAKAIMEQEANgIAIAUgACAAKAIAKAIQEQEANgIAIApBBGoiASAAIAAoAgAoAhQRAAAgBiABELYCIAEQxwUaIAEgACAAKAIAKAIYEQAAIAcgARCeBCABENAFGiAAIAAoAgAoAiQRAQALNgIAIApBEGokAAv1BwEKfyMAQRBrIhMkACACIAA2AgAgA0GABHEhFSAHQQJ0IRYDQCAUQQRGBEACfyANLQALQQd2BEAgDSgCBAwBCyANLQALQf8AcQtBAUsEQCATIA0QxQM2AgwgAiATQQxqQQEQoAQgDRDaAyACKAIAEKkENgIACyADQbABcSIDQRBHBEAgASADQSBGBH8gAigCAAUgAAs2AgALIBNBEGokAA8LAkACQAJAAkACQAJAIAggFGosAAAOBQABAwIEBQsgASACKAIANgIADAQLIAEgAigCADYCACAGQSAgBigCACgCLBEDACEHIAIgAigCACIPQQRqNgIAIA8gBzYCAAwDCwJ/IA0tAAtBB3YEQCANKAIEDAELIA0tAAtB/wBxC0UNAgJ/IA0tAAtBB3YEQCANKAIADAELIA0LKAIAIQcgAiACKAIAIg9BBGo2AgAgDyAHNgIADAILAn8gDC0AC0EHdgRAIAwoAgQMAQsgDC0AC0H/AHELRSEHIBVFDQEgBw0BIAIgDBDFAyAMENoDIAIoAgAQqQQ2AgAMAQsgAigCACEXIAQgFmoiBCEHA0ACQCAFIAdNDQAgBkHAACAHKAIAIAYoAgAoAgwRBABFDQAgB0EEaiEHDAELCyAOQQBKBEAgAigCACEPIA4hEANAAkAgBCAHTw0AIBBFDQAgEEEBayEQIAdBBGsiBygCACERIAIgD0EEaiISNgIAIA8gETYCACASIQ8MAQsLAkAgEEUEQEEAIREMAQsgBkEwIAYoAgAoAiwRAwAhESACKAIAIQ8LA0AgD0EEaiESIBBBAEoEQCAPIBE2AgAgEEEBayEQIBIhDwwBCwsgAiASNgIAIA8gCTYCAAsCQCAEIAdGBEAgBkEwIAYoAgAoAiwRAwAhDyACIAIoAgAiEEEEaiIHNgIAIBAgDzYCAAwBCwJ/IAstAAtBB3YEQCALKAIEDAELIAstAAtB/wBxCwR/An8gCy0AC0EHdgRAIAsoAgAMAQsgCwssAAAFQX8LIRFBACEPQQAhEANAIAQgB0cEQAJAIA8gEUcEQCAPIRIMAQsgAiACKAIAIhJBBGo2AgAgEiAKNgIAQQAhEgJ/IAstAAtBB3YEQCALKAIEDAELIAstAAtB/wBxCyAQQQFqIhBNBEAgDyERDAELAn8gCy0AC0EHdgRAIAsoAgAMAQsgCwsgEGotAABB/wBGBEBBfyERDAELAn8gCy0AC0EHdgRAIAsoAgAMAQsgCwsgEGosAAAhEQsgB0EEayIHKAIAIQ8gAiACKAIAIhhBBGo2AgAgGCAPNgIAIBJBAWohDwwBCwsgAigCACEHCyAXIAcQ6QMLIBRBAWohFAwACwALzwEBA38jAEEQayIFJAAjAEEgayIDJAAgA0EYaiAAIAEQmwUgA0EQaiADKAIYIAMoAhwgAhDSAiADKAIQIQQjAEEQayIBJAAgASAANgIMIAFBDGoiACAEIAAoAgAhBCMAQRBrIgAkACAAIAQ2AgwgACgCDCEEIABBEGokACAEa0ECdRCgBCEAIAFBEGokACADIAA2AgwgAyACIAMoAhQgAmtqNgIIIAUgAygCDDYCCCAFIAMoAgg2AgwgA0EgaiQAIAUoAgwhACAFQRBqJAAgAAuFBwEIfyMAQeADayIGJAAgBkHcA2oiCCADKAIcIgA2AgAgACAAKAIEQQFqNgIEBkAgCBCwAiEKAn8gBS0AC0EHdgRAIAUoAgQMAQsgBS0AC0H/AHELBH8CfyAFLQALQQd2BEAgBSgCAAwBCyAFCygCACAKQS0gCigCACgCLBEDAEYFQQALIQwGQCMAQRBrIgAkACAGQcQDaiIIQgA3AgAgCEEANgIIIABBEGokACMAQRBrIgAkACAGQbgDaiIHQgA3AgAgB0EANgIIIABBEGokACACIAwgBkHcA2ogBkHYA2ogBkHUA2ogBkHQA2ogCCAHIwBBEGsiAiQAIAZBrANqIgBCADcCACAAQQA2AgggAkEQaiQAIAAgBkGoA2oQpwQgBkHEATYCECAGQQhqQQAgBkEQaiICEL4CIQkGQAJAAn8CfyAFLQALQQd2BEAgBSgCBAwBCyAFLQALQf8AcQsgBigCqANKBEACfyAFLQALQQd2BEAgBSgCBAwBCyAFLQALQf8AcQshCyAGKAKoAyINAn8gBy0AC0EHdgRAIAcoAgQMAQsgBy0AC0H/AHELAn8gAC0AC0EHdgRAIAAoAgQMAQsgAC0AC0H/AHELIAsgDWtBAXRqampBAWoMAQsgBigCqAMCfyAALQALQQd2BEAgACgCBAwBCyAALQALQf8AcQsCfyAHLQALQQd2BEAgBygCBAwBCyAHLQALQf8AcQtqakECagsiC0HlAEkNACAJIAtBAnQQ8gEQlQMgCSgCACICDQAQpwUACyACIAZBBGogBiADKAIEAn8gBS0AC0EHdgRAIAUoAgAMAQsgBQsCfyAFLQALQQd2BEAgBSgCAAwBCyAFCwJ/IAUtAAtBB3YEQCAFKAIEDAELIAUtAAtB/wBxC0ECdGogCiAMIAZB2ANqIAYoAtQDIAYoAtADIAggByAAIAYoAqgDEKgEIAEgAiAGKAIEIAYoAgAgAyAEEN0DIQEZIAYkACAJEJYDCQALGSAGJAAgABDQBRogBxDQBRogCBDHBRoJAAsZIAYkACAGKALcAyIAIAAoAgRBAWsiATYCBCABQX9GBEAgACAAKAIAKAIIEQIACwkACyAJEJYDIAAQ0AUaIAcQ0AUaIAgQxwUaIAYoAtwDIgAgACgCBEEBayICNgIEIAJBf0YEQCAAIAAoAgAoAggRAgALIAZB4ANqJAAgAQsEAEF/CwoAIAAgBRCLBBoLnAIAIwBBEGsiAyQAAkAgBS0AC0EHdkUEQCAAIAUoAgg2AgggACAFKQIANwIADAELIAUoAgAhAiAFKAIEIQUjAEEQayIEJAACQAJAAkAgBUECSQRAIAAiASAALQALQYABcSAFcjoACyAAIAAtAAtB/wBxOgALDAELIAVB7////wNLDQEgBEEIaiAAIAVBAk8EfyAFQQRqQXxxIgEgAUEBayIBIAFBAkYbBUEBC0EBahCVBSAEKAIMGiAAIAQoAggiATYCACAAIAAoAghBgICAgHhxIAQoAgxB/////wdxcjYCCCAAIAAoAghBgICAgHhyNgIIIAAgBTYCBAsgASACIAVBAWoQrwIgBEEQaiQADAELEDMACwsgA0EQaiQACzMBAX8jACEBIABB+MIBNgIABkAgACgCCBCuA0cEQCAAKAIIEIMDCxkgASQAEPAFAAsgAAvwEgEEfyMAIgEhAkHMgANBADYCAEHIgANBmO4BNgIAQciAA0HwxQE2AgBByIADQai6ATYCAAZAIAFBEGsiASQAQdCAA0IANwMAIAFBADYCBEHYgANBADYCAEHYgQNBADoAACABQdCAAzYCACABKAIAIQAgAUEAOgAIIAEgADYCBAZAIwBBEGsiACQAQdCAAxCdBUEeSQRAEDcACyAAQQhqQeCAA0EeEJ4FQdSAAyAAKAIIIgM2AgBB0IADIAM2AgBB2IADIAMgACgCDEECdGo2AgBB2IADKAIAGkHQgAMoAgAaIABBEGokAEHQgANBHhCxBBkgASQAIAFBBGoQsgQJAAsgAUEEaiIAQQE6AAQgABCyBCABQRBqJAAGQEHggQNBpC8QMiEBQdSAAygCABpB0IADKAIAGkHQgAMQswRB2IADKAIAGkHUgAMoAgAaQdCAAygCABpBhP4CQQA2AgBBgP4CQZjuATYCAEGA/gJB8MUBNgIAQYD+AkHEzgE2AgAGQEHIgANBgP4CQbTyAhC0BBC1BEGM/gJBADYCAEGI/gJBmO4BNgIAQYj+AkHwxQE2AgBBiP4CQeTOATYCAEHIgANBiP4CQbzyAhC0BBC1BEGU/gJBADYCAEGQ/gJBmO4BNgIAQZD+AkHwxQE2AgBBnP4CQQA6AABBmP4CQQA2AgBBkP4CQby6ATYCAEGY/gJB8LoBNgIAQciAA0GQ/gJBgPQCELQEELUEQaT+AkEANgIAQaD+AkGY7gE2AgBBoP4CQfDFATYCAEGg/gJBqMYBNgIAQciAA0Gg/gJB+PMCELQEELUEQaz+AkEANgIAQaj+AkGY7gE2AgBBqP4CQfDFATYCAEGo/gJBvMcBNgIAQciAA0Go/gJBiPQCELQEELUEIwAhAEG0/gJBADYCAEGw/gJBmO4BNgIAQbD+AkHwxQE2AgBBsP4CQfjCATYCAAZAEK4DIQAZIAAkAAkAC0G4/gIgADYCAEHIgANBsP4CQZD0AhC0BBC1BEHE/gJBADYCAEHA/gJBmO4BNgIAQcD+AkHwxQE2AgBBwP4CQdDIATYCAEHIgANBwP4CQZj0AhC0BBC1BEHM/gJBADYCAEHI/gJBmO4BNgIAQcj+AkHwxQE2AgBByP4CQbjKATYCAEHIgANByP4CQaj0AhC0BBC1BEHU/gJBADYCAEHQ/gJBmO4BNgIAQdD+AkHwxQE2AgBB0P4CQcTJATYCAEHIgANB0P4CQaD0AhC0BBC1BEHc/gJBADYCAEHY/gJBmO4BNgIAQdj+AkHwxQE2AgBB2P4CQazLATYCAEHIgANB2P4CQbD0AhC0BBC1BEHk/gJBADYCAEHg/gJBmO4BNgIAQeD+AkHwxQE2AgBB6P4CQa7YADsBAEHg/gJBqMMBNgIAIwBBEGsiACQAQez+AkIANwIAQfT+AkEANgIAIABBEGokAEHIgANB4P4CQbj0AhC0BBC1BEH8/gJBADYCAEH4/gJBmO4BNgIAQfj+AkHwxQE2AgBBgP8CQq6AgIDABTcCAEH4/gJB0MMBNgIAIwBBEGsiACQAQYj/AkIANwIAQZD/AkEANgIAIABBEGokAEHIgANB+P4CQcD0AhC0BBC1BEGc/wJBADYCAEGY/wJBmO4BNgIAQZj/AkHwxQE2AgBBmP8CQYTPATYCAEHIgANBmP8CQcTyAhC0BBC1BEGk/wJBADYCAEGg/wJBmO4BNgIAQaD/AkHwxQE2AgBBoP8CQfjQATYCAEHIgANBoP8CQczyAhC0BBC1BEGs/wJBADYCAEGo/wJBmO4BNgIAQaj/AkHwxQE2AgBBqP8CQczSATYCAEHIgANBqP8CQdTyAhC0BBC1BEG0/wJBADYCAEGw/wJBmO4BNgIAQbD/AkHwxQE2AgBBsP8CQbTUATYCAEHIgANBsP8CQdzyAhC0BBC1BEG8/wJBADYCAEG4/wJBmO4BNgIAQbj/AkHwxQE2AgBBuP8CQYzcATYCAEHIgANBuP8CQYTzAhC0BBC1BEHE/wJBADYCAEHA/wJBmO4BNgIAQcD/AkHwxQE2AgBBwP8CQaDdATYCAEHIgANBwP8CQYzzAhC0BBC1BEHM/wJBADYCAEHI/wJBmO4BNgIAQcj/AkHwxQE2AgBByP8CQZTeATYCAEHIgANByP8CQZTzAhC0BBC1BEHU/wJBADYCAEHQ/wJBmO4BNgIAQdD/AkHwxQE2AgBB0P8CQYjfATYCAEHIgANB0P8CQZzzAhC0BBC1BEHc/wJBADYCAEHY/wJBmO4BNgIAQdj/AkHwxQE2AgBB2P8CQfzfATYCAEHIgANB2P8CQaTzAhC0BBC1BEHk/wJBADYCAEHg/wJBmO4BNgIAQeD/AkHwxQE2AgBB4P8CQaDhATYCAEHIgANB4P8CQazzAhC0BBC1BEHs/wJBADYCAEHo/wJBmO4BNgIAQej/AkHwxQE2AgBB6P8CQcTiATYCAEHIgANB6P8CQbTzAhC0BBC1BEH0/wJBADYCAEHw/wJBmO4BNgIAQfD/AkHwxQE2AgBB8P8CQejjATYCAEHIgANB8P8CQbzzAhC0BBC1BEH8/wJBADYCAEH4/wJBmO4BNgIAQfj/AkHwxQE2AgBBgIADQdDtATYCAEH4/wJB/NUBNgIAQYCAA0Gs1gE2AgBByIADQfj/AkHk8gIQtAQQtQRBjIADQQA2AgBBiIADQZjuATYCAEGIgANB8MUBNgIAQZCAA0H07QE2AgBBiIADQYTYATYCAEGQgANBtNgBNgIAQciAA0GIgANB7PICELQEELUEIwAhAAZAQZyAA0EANgIAQZiAA0GY7gE2AgBBmIADQfDFATYCAEGggAMQowUZIAAkAAkAC0GYgANB8NkBNgIAQciAA0GYgANB9PICELQEELUEIwAhAAZAQayAA0EANgIAQaiAA0GY7gE2AgBBqIADQfDFATYCAEGwgAMQowUZIAAkAAkAC0GogANBjNsBNgIAQciAA0GogANB/PICELQEELUEQbyAA0EANgIAQbiAA0GY7gE2AgBBuIADQfDFATYCAEG4gANBjOUBNgIAQciAA0G4gANBxPMCELQEELUEQcSAA0EANgIAQcCAA0GY7gE2AgBBwIADQfDFATYCAEHAgANBhOYBNgIAQciAA0HAgANBzPMCELQEELUEGSACJAAgARDHBRoJAAsZIAIkAEHQgAMQsAQJAAsZIAIkAAkACwsxAQJ/IwBBEGsiASQABkAgAUEMaiICIAA2AgAgAhC2BBkgASQAEPAFAAsgAUEQaiQAC3YBAX8jAEEQayICJAAgAiAANgIEIAIgACgCBCIANgIIIAIgACABQQJ0ajYCDCACKAIIIQEgAigCDCEAA0ACQCAAIAFHBEAgAUEANgIADAELIAIoAgQgAigCCDYCBCACQRBqJAAPCyACIAFBBGoiATYCCAwACwALIgEBfyAALQAERQRAIwAhAQZAIAAQtgQZIAEkABDwBQALCwsMACAAIAAoAgAQoAUL6gEBBX8jAEEgayIBJAAgAUEANgIQIAFBxgE2AgwgASABKQIMNwMAIAFBFGoiAiABKQIANwIEIAIgADYCACMAQRBrIgMkACAAKAIAQX9HBEAgA0EMaiIFIAI2AgAgA0EIaiIEIAU2AgAjACECA0AgACgCAEEBRg0ACyAAKAIARQRAAkAgAEEBNgIABkAgBBC/BCAAQX82AgAMAQcAIQEgAiQAIAEQ6QUaBkAgAEEANgIAEOsFGSACJAAGQBDqBRkgAiQAEPAFAAsJAAsACwALCwsgA0EQaiQAIAAoAgQhACABQSBqJAAgAEEBawuzAgEDfyMAQRBrIgUkACABIAEoAgRBAWo2AgQjAEEQayIDJAAgAyABNgIMIAVBDGoiASADKAIMNgIAIANBEGokACACIABBCGoiACgCBCAAKAIAa0ECdU8EQAZAAkAgACgCBCAAKAIAa0ECdSIEIAJBAWoiA0kEQCAAIAMgBGsQugQMAQsgAyAESQRAIAAoAgQaIAAoAgAhBCAAIANBAnQgBGoQoAUgACgCCBogACgCBBogACgCABoLCxkgBSQAIAEQtwQJAAsLIAAoAgAgAkECdGooAgAEQCAAKAIAIAJBAnRqKAIAIgMgAygCBEEBayIENgIEIARBf0YEQCADIAMoAgAoAggRAgALCyABKAIAIQMgAUEANgIAIAAoAgAgAkECdGogAzYCACABELcEIAVBEGokAAtIAQF/IAAoAgAiASgCBBogASgCCBogASgCABogASgCAARAIAEQswQgACgCACIAQRBqIAAoAgAgACgCCCAAKAIAa0ECdRCfBQsLOwEBfyAAKAIAIQEgAEEANgIAIAEEQCABIAEoAgRBAWsiADYCBCAAQX9GBEAgASABKAIAKAIIEQIACwsLiAEBBH8gAEGougE2AgAgAEEIaiEBA0AgAiABKAIEIAEoAgBrQQJ1SQRAIAEoAgAgAkECdGooAgAEQCABKAIAIAJBAnRqKAIAIgMgAygCBEEBayIENgIEIARBf0YEQCADIAMoAgAoAggRAgALCyACQQFqIQIMAQsLIABBmAFqEMcFGiABELAEIAALDQAgABC4BBogABDzAQvGBwEKfyMAQSBrIgkkAAJAIAEgACgCCCAAKAIEa0ECdU0EQCAAIAEQsQQMAQsgAEEQaiEHBkAGQCAJQQxqIQMCfyABIAAoAgQgACgCAGtBAnVqIQQjAEEQayICJAAgAiAENgIMIAQgACIFEJ0FIgBNBEAgBSgCCCAFKAIAa0ECdSIEIABBAXZJBEAgAiAEQQF0NgIIIwBBEGsiACQAIAJBCGoiBCgCACACQQxqIgYoAgBJIQggAEEQaiQAIAYgBCAIGygCACEACyACQRBqJAAgAAwBCxA3AAshBCAFKAIEIAUoAgBrQQJ1IQZBACECIwBBEGsiACQAIABBADYCDCADQQA2AgwgAyAHNgIQIAQEfyAAQQRqIAMoAhAgBBCeBSAAKAIEIQIgACgCCAVBAAshBCADIAI2AgAgAyACIAZBAnRqIgc2AgggAyAHNgIEIAMgAiAEQQJ0ajYCDCAAQRBqJAAYAiMAQRBrIgAkACAAIAMoAgg2AgQgAygCCCECIAAgA0EIajYCDCAAIAIgAUECdGo2AgggACgCBCECA0ACQAJAIAAoAgggAkcEQCADKAIQGiAAKAIEQQA2AgAMAQsgACgCDCAAKAIENgIAIABBEGokAAwBCyAAIAAoAgRBBGoiAjYCBAwBCwsjAEEQayIBJAAgBSgCCBogBSgCABogASAFKAIENgIIIAEgBSgCADYCBCABIAMoAgQ2AgAgASgCCCEEIAEoAgQhBiABKAIAIQgjAEEQayIHJAAjAEEQayICJAAjAEEgayIAJAAgACAGNgIYIAAgBDYCHCAAIAg2AhQgACgCGCIEIQYgACgCFCAEIAAoAhwiCGtqIQojAEEQayIEJAAgCiAGIAggBmsiBhDTASELIAQgCDYCDCAEIAYgC2o2AgggACAEKAIMNgIMIAAgBCgCCDYCECAEQRBqJAAgACAKIAAoAhQiBGsgBGo2AgwgAiAAKAIYNgIIIAIgACgCDDYCDCAAQSBqJAAgAiACKAIINgIEIAIgAigCDDYCACAHIAIoAgQ2AgggByACKAIANgIMIAJBEGokACAHKAIMIQAgB0EQaiQAIAEgADYCDCADIAEoAgw2AgQgBSgCACEAIAUgAygCBDYCACADIAA2AgQgBSgCBCEAIAUgAygCCDYCBCADIAA2AgggBSgCCCEAIAUgAygCDDYCCCADIAA2AgwgAyADKAIENgIAIAUoAgQaIAUoAgAaIAUoAggaIAUoAgAaIAFBEGokABkgCSQAIAMQoQUJAAsgAxChBQsgCUEgaiQACy8AIAEgAEEIaiIAKAIEIAAoAgBrQQJ1SQR/IAAoAgAgAUECdGooAgBBAEcFQQALC7IBAQF/IwAhAQZAAn9B8PMCLQAABEBB7PMCKAIADAELQejzAgJ/QeTzAi0AAARAQeDzAigCAAwBCxCvBEHc8wJByIADNgIAQeTzAkEBOgAAQeDzAkHc8wI2AgBB3PMCCygCACIBNgIAIAEgASgCBEEBajYCBEHw8wJBAToAAEHs8wJB6PMCNgIAQejzAgshARkgASQAEPAFAAsgACABKAIAIgA2AgAgACAAKAIEQQFqNgIECxwAIABB9PMCQfTzAigCAEEBaiIANgIAIAA2AgQLDwAgACAAKAIAKAIEEQIAC0ABAn8gACgCACgCACIAKAIAIAAoAggiAkEBdWohASAAKAIEIQAgASACQQFxBH8gASgCACAAaigCAAUgAAsRAgALJQBBACEAIAJB/wBNBH8gAkECdEHwugFqKAIAIAFxQQBHBUEACwtJAQF/A0AgASACRkUEQEEAIQAgAyABKAIAIgRB/wBNBH8gBEECdEHwugFqKAIABUEACzYCACADQQRqIQMgAUEEaiEBDAELCyACC0AAA0ACQCACIANHBH8gAigCACIAQf8ASw0BIABBAnRB8LoBaigCACABcUUNASACBSADCw8LIAJBBGohAgwACwALQQACQANAIAIgA0YNAQJAIAIoAgAiAEH/AEsNACAAQQJ0QfC6AWooAgAgAXFFDQAgAkEEaiECDAELCyACIQMLIAMLHgAgAUH/AE0Ef0GYoAEoAgAgAUECdGooAgAFIAELC0EAA0AgASACRwRAIAEgASgCACIAQf8ATQR/QZigASgCACABKAIAQQJ0aigCAAUgAAs2AgAgAUEEaiEBDAELCyACCx4AIAFB/wBNBH9BoKwBKAIAIAFBAnRqKAIABSABCwtBAANAIAEgAkcEQCABIAEoAgAiAEH/AE0Ef0GgrAEoAgAgASgCAEECdGooAgAFIAALNgIAIAFBBGohAQwBCwsgAgsqAANAIAEgAkZFBEAgAyABLAAANgIAIANBBGohAyABQQFqIQEMAQsLIAILDgAgASACIAFBgAFJG8ALNQADQCABIAJGRQRAIAQgASgCACIAIAMgAEGAAUkbOgAAIARBAWohBCABQQRqIQEMAQsLIAILKQEBfyAAQby6ATYCAAJAIAAoAggiAUUNACAALQAMRQ0AIAEQ8wELIAALDQAgABDLBBogABDzAQsiACABQQBOBH9BmKABKAIAIAFB/wFxQQJ0aigCAAUgAQvAC0AAA0AgASACRwRAIAEgASwAACIAQQBOBH9BmKABKAIAIAEsAABBAnRqKAIABSAACzoAACABQQFqIQEMAQsLIAILIgAgAUEATgR/QaCsASgCACABQf8BcUECdGooAgAFIAELwAtAAANAIAEgAkcEQCABIAEsAAAiAEEATgR/QaCsASgCACABLAAAQQJ0aigCAAUgAAs6AAAgAUEBaiEBDAELCyACCyoAA0AgASACRkUEQCADIAEtAAA6AAAgA0EBaiEDIAFB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kgIgACACIANqIgEQkwQgBUEAOgAPIAEgBGogBS0ADzoAAAwBCyAAIAQgAiADaiAEayADIANBACACIAEQxgULIAVBEGokACAAC4ECAQR/An8gARDjASECIwBBEGsiBSQAAn8gAC0AC0EHdgRAIAAoAgQMAQsgAC0AC0H/AHELIgRBAE8EQAJAIAIgAC0AC0EHdgR/IAAoAghB/////wdxQQFrBUEKCyIDIARrTQRAIAJFDQECfyAALQALQQd2BEAgACgCAAwBCyAACyIDIAQEfyACIANqIAMgBBDFBSABIAJBACADIARqIAFLG0EAIAEgA08bagUgAQsgAhDFBSAAIAIgBGoiARCTBCAFQQA6AA8gASADaiAFLQAPOgAADAELIAAgAyACIARqIANrIARBAEEAIAIgARDGBQsgBUEQaiQAIAAMAQsQogUACwv5AQEDfyMAQRBrIgIkACACIAE6AA8CQAJAAn8gAC0AC0EHdiIERQRAQQohASAALQALQf8AcQwBCyAAKAIIQf////8HcUEBayEBIAAoAgQLIgMgAUYEQCAAIAFBASABIAEQyAUCfyAALQALQQd2BEAgACgCAAwBC0EACxoMAQsCfyAALQALQQd2BEAgACgCAAwBC0EACxogBA0AIAAiASADQQFqIAAtAAtBgAFxcjoACyAAIAAtAAtB/wBxOgALDAELIAAoAgAhASAAIANBAWo2AgQLIAEgA2oiACACLQAPOgAAIAJBADoADiAAIAItAA46AAEgAkEQaiQACw4AIAAgASABEOMBEMsFC58DAQV/IwBBEGsiCCQAIAIgAUF/c0Hv////A2pNBEACfyAALQALQQd2BEAgACgCAAwBCyAACyEJIAhBBGogACABQef///8BSQR/IAggAUEBdDYCDCAIIAEgAmo2AgQjAEEQayICJAAgCEEEaiIKKAIAIAhBDGoiCygCAEkhDCACQRBqJAAgCyAKIAwbKAIAIgJBAk8EfyACQQRqQXxxIgIgAkEBayICIAJBAkYbBUEBC0EBagVB7////wMLEJUFIAgoAgQhAiAIKAIIGiAEBEAgAiAJIAQQrwILIAYEQCAEQQJ0IAJqIAcgBhCvAgsgAyAEIAVqIgprIQcgAyAKRwRAIARBAnQiAyACaiAGQQJ0aiADIAlqIAVBAnRqIAcQrwILIAFBAWoiAUECRwRA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AEdxRQRAQQAhAiMAQRBrIgMkAAJAAkACQCAAKAIAIgEgACgCBEcEfyABLQAABUEAC8AiAUHHAEcEQCABQdQARw0DAkACQAJAAkACQAJAAkACQAJAAkACQCAAKAIEIAAoAgAiAWtBAUsEfyABLQABBUEAC8AiAUHBAGsOCQEKBgoKCgoIBAALIAFB0wBrDgUEAgkBBggLIAAgACgCAEECajYCACADIAAQrAYiAjYCBCACRQ0LIwBBEGsiASQAIABBmANqQRQQxwYhACABQQhqQfDEABCdBiECIAMoAgQhBCABIAIpAgA3AwAgACABIAQQyAYhAiABQRBqJAAMDAsgACAAKAIAQQJqNgIAIAMgABCjBiICNgIEIAJFDQojAEEQayIBJAAgAEGYA2pBFBDHBiEAIAFBCGpBiMYAEJ0GIQIgAygCBCEEIAEgAikCADcDACAAIAEgBBDIBiECIAFBEGokAAwLCyAAIAAoAgBBAmo2AgAgAyAAEKMGIgI2AgQgAkUNCSMAQRBrIgEkACAAQZgDakEUEMcGIQAgAUEIakGoxgAQnQYhAiADKAIEIQQgASACKQIANwMAIAAgASAEEMgGIQIgAUEQaiQADAoLIAAgACgCAEECajYCACADIAAQowYiAjYCBCACRQ0IIwBBEGsiASQAIABBmANqQRQQxwYhACABQQhqQY/FABCdBiECIAMoAgQhBCABIAIpAgA3AwAgACABIAQQyAYhAiABQRBqJAAMCQsgACAAKAIAQQJqNgIAIAMgABCjBiICNgIEIAJFDQcjAEEQayIBJAAgAEGYA2pBFBDHBiEAIAFBCGpB6MUAEJ0GIQIgAygCBCEEIAEgAikCADcDACAAIAEgBBDIBiECIAFBEGokAAwICyAAIAAoAgBBAmo2AgAgAyAAEKMGIgE2AgwgAUUNByADQQRqIABBARChBiADKAIEIAMoAghGDQcgAEHfABCgBkUNByADIAAQowYiAjYCBCACRQ0GIABBmANqQRAQxwYhACADKAIEIQIgAygCDCEBIABBFUEAQQFBAUEBEMkGIgAgATYCDCAAIAI2AgggAEHkkwI2AgAgACECDAcLIAAgACgCAEECajYCACADIABBABCqBiIBNgIEIAFFDQYgAEGdxQAgA0EEahCiBiECDAYLIAAgACgCAEECajYCACADIABBABCqBiIB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EQAAIAAvAAVBwAFxQcAARwRAIAAgASAAKAIAKAIUEQAACyACQRBqJAALQAEBfyABKAIEIAEoAgBrIgIEQCAAIAIQpAYgACgCACAAKAIEaiABKAIAIAIQ0gEaIAAgACgCBCACajYCBAsgAAsJACAAQgA3AgALtgEBAn8jAEEgayICJAAgAiACQRhqQfvFABCdBikCADcDCCABIAJBCGoQywYhASAAKAIIIgMgASADKAIAKAIQEQAAIAMvAAVBwAFxQcAARwRAIAMgASADKAIAKAIUEQAACyACIAJBEGpB3zwQnQYpAgA3AwAgASACEMsGIQEgACgCDCIAIAEgACgCACgCEBEAACAALwAFQcABcUHAAEcEQCAAIAEgACgCACgCFBEAAAsgAkEgaiQAC6ABAQN/IAFBADYCAAJAIAAoAgAiAiAAKAIERwR/IAItAAAFQQALwEE6a0H/AXFB9gFJIgMNAANAIAAoAgAiAiAAKAIERwR/IAItAAAFQQALwEEwa0H/AXFBCUsNASABIARBCmw2AgAgASAAKAIAIgIgACgCBEYEf0EABSAAIAJBAWo2AgAgAi0AAAvAIAEoAgBqQTBrIgQ2AgAMAAsACyADCyYAIABBB0EAQQFBAUEBEMkGIgBB2JQCNgIAIAAgASkCADcCCCAACzECAX8BfiMAQRBrIgIkACACIAApAggiAzcDACACIAM3AwggASACEMsGGiACQRBqJAALDAAgACABKQIINwIAC5MBAQF/QQAgACgCCCICBH8gAiABIAIoAgAoAhARAAAgAi8ABUHAAXFBwABHBEAgAiABIAIoAgAoAhQRAAALIAAoAghFBUEBCyAALQAQIgIbRQRAIAFBOkEuIAIbEJwGGgsgACgCDCIAIAEgACgCACgCEBEAACAALwAFQcABcUHAAEcEQCAAIAEgACgCACgCFBEAAAsLbAEBfyMAQRBrIgEkACABQQA2AgwgAEHyABCgBgRAIAEgASgCDEEEcjYCDAsgAEHWABCgBgRAIAEgASgCDEECcjYCDAsgAEHLABCgBgRAIAEgASgCDEEBcjYCDAsgASgCDCEAIAFBEGokACAAC44DAQN/IwBBEGsiASQAAkAgAEHTABCgBkUNACAAKAIAIgIgACgCBEcEfyACLQAABUEAC8AiAkHhAGtB/wFxQRlNBEACQAJAAkACQAJA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AAAgACgCDCABEN8GRQRAIAIgAkEIakGtzAAQnQYpAgA3AwAgASACEMsGGgsgAkEQaiQAC0sBAX8gACgCCCICIAEgAigCACgCEBEAACACLwAFQcABcUHAAEcEQCACIAEgAigCACgCFBEAAAsgACgCDCIAIAEgACgCACgCFBEAAAueAQECfyMAQTBrIgIkACACQShqIgMgAUEUajYCACADIAEoAhQ2AgQgAUEANgIUIAIgAkEgakHkNhCdBikCADcDEAZAIABBDGogASACQRBqEMsGIgAQgQcZIAIkACADKAIAIAMoAgQ2AgAJAAsgAiACQRhqQZXHABCdBikCADcDCCAAIAJBCGoQywYaIAMoAgAgAygCBDYCACACQTBqJAALQwEBfyMAQRBrIgIkACAAKAIIIgAgASAAKAIAKAIQEQAAIAIgAkEIakGKPBCdBikCADcDACABIAIQywYaIAJBEGokAAsWACAAKAIIIgAgASAAKAIAKAIUEQAAC3MCAX8BfiMAQTBrIgIkACACIAJBKGpByysQnQYpAgA3AxAgASACQRBqEMsGIQEgAiAAKQIYIgM3AwggAiADNwMgIAEgAkEIahDLBiEBIAIgAkEYakHWPxCdBikCADcDACAAIAEgAhDLBhCdByACQTBqJAALvwEBA38jAEEwayICJAAgAEEIaiIEKAIEBEAgAkEoaiIDIAFBFGo2AgAgAyABKAIUNgIEIAFBADYCFCACIAJBIGpBgTcQnQYpAgA3AxAGQCAEIAEgAkEQahDLBiIEEIEHGSACJAAgAygCACADKAIENgIACQALIAIgAkEYakGyNRCdBikCADcDCCAEIAJBCGoQywYaIAMoAgAgAygCBDYCAAsgAUEoEPkGIABBEGogARCBByABQSkQ+wYgAkEwaiQAC2UBAX8jAEEgayICJAAgAiACQRhqQZAsEJ0GKQIANwMIIAEgAkEIahDLBiEBIAAoAggiAC0ABEEwRgRAIAAgARCdBwsgAiACQRBqQbYJEJ0GKQIANwMAIAEgAhDLBhogAkEgaiQAC8sBAgJ/AX4jAEEwayICJAAgAUEoEPkGIAAoAggiAyABIAMoAgAoAhARAAAgAy8ABUHAAXFBwABHBEAgAyABIAMoAgAoAhQRAAALIAFBKRD7BgJAIABBDGoiACgCAC0AAEHuAEYEQCACIAJBKGpB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mANqQQwQxwYhACABKAIAIQEgAEEEQQBBAUEBQQEQyQYiACABNgIIIABBhL0CNgIAIAALRAEBfyMAQRBrIgIkACACIAJBCGpBtAkQnQYpAgA3AwAgASACEMsGIQEgACgCCCIAIAEgACgCACgCEBEAACACQRBqJAALZAEBfyMAQRBrIgIkACACIAJBCGpBzcQAEJ0GKQIANwMAIAEgAhDLBiEBIAAoAggiACABIAAoAgAoAhARAAAgAC8ABUHAAXFBwABHBEAgACABIAAoAgAoAhQRAAALIAJBEGokAAtkAQF/IwBBEGsiAiQAIAIgAkEIakGjzAAQnQYpAgA3AwAgASACEMsGIQEgACgCCCIAIAEgACgCACgCEBEAACAALwAFQcABcUHAAEcEQCAAIAEgACgCACgCFBEAAAsgAkEQaiQAC2MBAX8jAEEQayICJAAgAiACQQhqQZk3EJ0GKQIANwMAIAEgAhDLBiEBIAAoAggiACABIAAoAgAoAhARAAAgAC8ABUHAAXFBwABHBEAgACABIAAoAgAoAhQRAAALIAJBEGokAAsWACAAIAEoAggiACAAKAIAKAIYEQAACyMAIAAgAkEAQQFBAUEBEMkGIgAgATYCCCAAQZDBAjYCACAAC08BAX8jAEEgayICJAAgAiACQRhqQZY3EJ0GKQIANwMIIAEgAkEIahCgByEBIAJBEGogABC5ByACIAIpAhA3AwAgASACEKAHGiACQSBqJAALbwEBfyMAQSBrIgIkACAAIAEQugcCQCABKAIIQQFLBEAgAiACQRhqQdYrEJ0GKQIANwMIIAAgAkEIahCpBkUNASACQRBqIABBBhDtBiAAIAIpAxA3AgALIAJBIGokAA8LQa0/QcshQaEMQcgkEBUACxgAIAAgASgCCEECdEGE2gJqKAIAEJ0GGgvMAQEBfyMAQdAAayICJAAgAiACQcgAakGWNxCdBikCADcDICABIAJBIGoQoAchASACQUBrIAAgACgCACgCGBEAACACIAIpAkA3AxggASACQRhqEKAHIQEgACgCCEEBSwRAIAIgAkE4akGLMxCdBikCADcDECABIAJBEGoQoAchASAAKAIIQQJGBEAgAiACQTBqQakzEJ0GKQIANwMIIAEgAkEIahCgBxoLIAIgAkEoakGyNRCdBikCADcDACABIAIQoAcaCyACQdAA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ICAwWDB4AC0wWCswWCAwWC0weCsweCAweAB0dwB0aHJvdwBfX25leHRfcHJpbWUgb3ZlcmZsb3cAb3BlcmF0b3IgbmV3AER3AE5vdgBEdgBUaHUAVHUAQXVndXN0ACBjb25zdABOb3QgZW5vdWdoIG1lbW9yeTogbG9hZEluZGV4IGZhaWxlZCB0byBhbGxvY2F0ZSBsaW5rbGlzdABOb3QgZW5vdWdoIG1lbW9yeTogYWRkUG9pbnQgZmFpbGVkIHRvIGFsbG9jYXRlIGxpbmtsaXN0AFRoZSBuZXdseSBpbnNlcnRlZCBlbGVtZW50IHNob3VsZCBoYXZlIGJsYW5rIGxpbmsgbGlzdAB+U2NvcGVkVGVtcGxhdGVQYXJhbUxpc3QAY29uc3RfY2FzdAByZWludGVycHJldF9jYXN0AHN0ZDo6YmFkX2Nhc3QAc3RhdGljX2Nhc3QAZHluYW1pY19jYXN0AHVuc2lnbmVkIHNob3J0AG5vZXhjZXB0AGsgPD0gY3VyX2VsZW1lbnRfY291bnQAaW50ZXJuYWxJZCA8IGN1cl9lbGVtZW50X2NvdW50AGdldEN1cnJlbnRDb3VudABnZXRQb2ludABub3JtYWxpemVQb2ludAByZW1vdmVQb2ludABhZGRQb2ludAB1bnNpZ25lZCBpbnQAX0JpdEludABUaGUgbnVtYmVyIG9mIGVsZW1lbnRzIGV4Y2VlZHMgdGhlIHNwZWNpZmllZCBsaW1pdABvcGVyYXRvciBjb19hd2FpdAB1bmNhdWdodABzdHJ1Y3QAIHJlc3RyaWN0AG9iamNfb2JqZWN0AE9jdABwb3NpeF9zdGF0AGZsb2F0AF9GbG9hdABTYXQAc3RkOjpudWxscHRyX3QAd2NoYXJfdABjaGFyOF90AGNoYXIxNl90AHVpbnQ2NF90AGNoYXIzMl90AFV0AFR0AFN0AGNoZWNrRmlsZUV4aXN0cwBOb3QgZW5vdWdoIG1lbW9yeTogbG9hZEluZGV4IGZhaWxlZCB0byBhbGxvY2F0ZSBsaW5rbGlzdHMATm90IGVub3VnaCBtZW1vcnk6IEhpZXJhcmNoaWNhbE5TVyBmYWlsZWQgdG8gYWxsb2NhdGUgbGlua2xpc3RzAFBhcnNlci0+VGVtcGxhdGVQYXJhbXMuc2l6ZSgpID49IE9sZE51bVRlbXBsYXRlUGFyYW1MaXN0cwBhZGRQb2ludHMAQ2Fubm90IHJlc2l6ZSwgbWF4IGVsZW1lbnQgaXMgbGVzcyB0aGFuIHRoZSBjdXJyZW50IG51bWJlciBvZiBlbGVtZW50cwBnZXRNYXhFbGVtZW50cwBuZWlnaGJvcnMATm90IGVub3VnaCBtZW1vcnk6IHJlc2l6ZUluZGV4IGZhaWxlZCB0byBhbGxvY2F0ZSBvdGhlciBsYXllcnMAZ2V0TnVtRGltZW5zaW9ucwBtYXJrRGVsZXRlSXRlbXMAYWRkSXRlbXMAZ2V0RGVsZXRlZExhYmVscwBnZXRVc2VkTGFiZWxzAHRoaXMAc2V0RGVidWdMb2dzAGRpc3RhbmNlcwBUcwB0ZXJtaW5hdGluZyBkdWUgdG8gJXMgZXhjZXB0aW9uIG9mIHR5cGUgJXMAdGVybWluYXRpbmcgZHVlIHRvICVzIGV4Y2VwdGlvbiBvZiB0eXBlICVzOiAlcwBmaWxlc3lzdGVtIGVycm9yOiAlcwBudWxscHRyAHNyAEFwcgBSZXBsYWNlbWVudCBvZiBkZWxldGVkIGVsZW1lbnRzIGlzIGRpc2FibGVkIGluIGNvbnN0cnVjdG9yAEN1c3RvbUZpbHRlckZ1bmN0b3IAdmVjdG9yAG9wZXJhdG9yAGFsbG9jYXRvcgB1bnNwZWNpZmllZCBpb3N0cmVhbV9jYXRlZ29yeSBlcnJvcgBtb25leV9nZXQgZXJyb3IATGV2ZWwgZXJyb3IAY2FuZCBlcnJvcgBOb3QgZW5vdWdoIG1lbW9yeTogcmVzaXplSW5kZXggZmFpbGVkIHRvIGFsbG9jYXRlIGJhc2UgbGF5ZXIAcmVhZEVuY29kZWRQb2ludGVyAEJhZCB2YWx1ZSBvZiBzel9saW5rX2xpc3Rfb3RoZXIARW1zY3JpcHRlbkZpbGVTeXN0ZW1NYW5hZ2VyAE9jdG9iZXIATm92ZW1iZXIAU2VwdGVtYmVyAERlY2VtYmVyAHVuc2lnbmVkIGNoYXIAaW9zX2Jhc2U6OmNsZWFyAE1hcgBzcAAvb3B0L2hvbWVicmV3L0NlbGxhci9lbXNjcmlwdGVuLzMuMS4zNS9saWJleGVjL3N5c3RlbS9saWIvbGliY3h4YWJpL3NyYy9jeGFfcGVyc29uYWxpdHkuY3BwAC9vcHQvaG9tZWJyZXcvQ2VsbGFyL2Vtc2NyaXB0ZW4vMy4xLjM1L2xpYmV4ZWMvc3lzdGVtL2xpYi9saWJjeHhhYmkvc3JjL2N4YV9kZW1hbmdsZS5jcHAAb3AAZnAAU2VwAFRwACVJOiVNOiVTICVwAGF1dG8Ab2JqY3Byb3RvAHNvAERvAFN1bgBKdW4AeXB0bgBQb3NzaWJsZSBtZW1vcnkgY29ycnVwdGlvbgBzdGQ6OmV4Y2VwdGlvbgB0ZXJtaW5hdGluZyBkdWUgdG8gJXMgZm9yZWlnbiBleGNlcHRpb24AdGVybWluYXRlX2hhbmRsZXIgdW5leHBlY3RlZGx5IHRocmV3IGFuIGV4Y2VwdGlvbgBDdXJyZW50UG9zaXRpb24AdW5pb24ATW9uAHNlYXJjaEtubgBkbgBuYW4ASmFuAFRuAERuAGVudW0Ac3lzdGVtAGluaXRpYWxpemVGaWxlU3lzdGVtAHBhcnNlVGVtcGxhdGVQYXJhbQBiYXNpY19pb3N0cmVhbQBiYXNpY19vc3RyZWFtAGJhc2ljX2lzdHJlYW0ASnVsAHRsAGJvb2wAZ2V0U3ltYm9sAHVsbABjYWxsAEFwcmlsAExldmVsIG9mIGl0ZW0gdG8gYmUgdXBkYXRlZCBjYW5ub3QgYmUgYmlnZ2VyIHRoYW4gbWF4IGxldmVsAFRyeWluZyB0byBtYWtlIGEgbGluayBvbiBhIG5vbi1leGlzdGVudCBsZXZlbABlbXNjcmlwdGVuOjp2YWwAc3RyaW5nIGxpdGVyYWwAdW5tYXJrRGVsZXRlZEludGVybmFsAFVsAHBvcF9iYWNrAGRyb3BCYWNrAEZyaQBwaQBsaQBiYWRfYXJyYXlfbmV3X2xlbmd0aABmYWlsZWQgdG8gZGV0ZXJtaW5lIGF0dHJpYnV0ZXMgZm9yIHRoZSBzcGVjaWZpZWQgcGF0aABzZXRFZlNlYXJjaABnZXRFZlNlYXJjaABCcnV0ZWZvcmNlU2VhcmNoAE1hcmNoAC9vcHQvaG9tZWJyZXcvQ2VsbGFyL2Vtc2NyaXB0ZW4vMy4xLjM1L2xpYmV4ZWMvc3lzdGVtL2xpYi9saWJjeHhhYmkvc3JjL2RlbWFuZ2xlL1V0aWxpdHkuaAAvb3B0L2hvbWVicmV3L0NlbGxhci9lbXNjcmlwdGVuLzMuMS4zNS9saWJleGVjL3N5c3RlbS9saWIvbGliY3h4YWJpL3NyYy9kZW1hbmdsZS9TdHJpbmdWaWV3LmgALi8uL3NyYy9obnN3bGliL2huc3dhbGcuaAAvb3B0L2hvbWVicmV3L0NlbGxhci9lbXNjcmlwdGVuLzMuMS4zNS9saWJleGVjL3N5c3RlbS9saWIvbGliY3h4YWJpL3NyYy9kZW1hbmdsZS9JdGFuaXVtRGVtYW5nbGUuaAAuLy4vc3JjL2huc3dsaWIvYnJ1dGVmb3JjZS5oAEF1ZwB1bnNpZ25lZCBsb25nIGxvbmcAdW5zaWduZWQgbG9uZwB0ZXJtaW5hdGluZwBzdGQ6OndzdHJpbmcAYmFzaWNfc3RyaW5nAHN0ZDo6c3RyaW5nAHN0ZDo6dTE2c3RyaW5nAHN0ZDo6dTMyc3RyaW5nAF9fdXVpZG9mAGluZgBUcnlpbmcgdG8gY29ubmVjdCBhbiBlbGVtZW50IHRvIGl0c2VsZgBoYWxmACVhZgAlLjBMZgAlTGYAdHJ1ZQBUdWUAb3BlcmF0b3IgZGVsZXRlAHVubWFya0RlbGV0ZQBmYWxzZQBkZWNsdHlwZQBKdW5lAGNvc2luZQBnZXRCYXNlTmFtZQAgdm9sYXRpbGUAQ2Fubm90IG9wZW4gZmlsZQBfX2N4YV9kZW1hbmdsZQBsb25nIGRvdWJsZQBfYmxvY2tfaW52b2tlAEZvcndhcmRSZWYtPmdldEtpbmQoKSA9PSBOb2RlOjpLRm9yd2FyZFRlbXBsYXRlUmVmZXJlbmNlAGRpc3RhbmNlAElubmVyUHJvZHVjdFNwYWNlAEwyU3BhY2UAVGUAc3RkAExhYmVsIG5vdCBmb3VuZAB2b2lkAEF1dG9TYXZlIG5vdCBlbmFibGVkIG9yIG5vdCBpbml0aWFsaXplZABpc0luZGV4SW5pdGlhbGl6ZWQAaXNJbml0aWFsaXplZABJbmRleCBzZWVtcyB0byBiZSBjb3JydXB0ZWQgb3IgdW5zdXBwb3J0ZWQAbG9jYWxlIG5vdCBzdXBwb3J0ZWQAVGhlIHJlcXVlc3RlZCB0byBkZWxldGUgZWxlbWVudCBpcyBhbHJlYWR5IGRlbGV0ZWQAVGhlIHJlcXVlc3RlZCB0byB1bmRlbGV0ZSBlbGVtZW50IGlzIG5vdCBkZWxldGVkAHVuZXhwZWN0ZWQAdGVybWluYXRlX2hhbmRsZXIgdW5leHBlY3RlZGx5IHJldHVybmVkACd1bm5hbWVkAG11dGV4IGxvY2sgZmFpbGVkAFRoZSBtYXhpbXVtIG51bWJlciBvZiBlbGVtZW50cyBoYXMgYmVlbiByZWFjaGVkAGUuIElEQkZTIGhhcyBzeW5jZWQAYy4gSURCRlMgaGFzIHN5bmNlZABpc1N5bmNlZABzZXRJZGJmc1N5bmNlZABXZWQAVW5rbm93biBlcnJvciAlZABzdGQ6OmJhZF9hbGxvYwBtYwBTaG91bGQgYmUgbm90IGJlIG1vcmUgdGhhbiBNXyBjYW5kaWRhdGVzIHJldHVybmVkIGJ5IHRoZSBoZXVyaXN0aWMAZ2VuZXJpYwBEZWMAd2IAcmIARmViAHNjYW5fZWhfdGFiAFViAHcrYgByK2IAYStiAHJ3YQBOb3QgZW5vdWdoIG1lbW9yeTogbG9hZEluZGV4IGZhaWxlZCB0byBhbGxvY2F0ZSBkYXRhAE5vdCBlbm91Z2ggbWVtb3J5OiBCcnV0ZWZvcmNlU2VhcmNoIGZhaWxlZCB0byBhbGxvY2F0ZSBkYXRhACdsYW1iZGEAJWEAYmFzaWNfAG9wZXJhdG9yXgBvcGVyYXRvciBuZXdbXQBvcGVyYXRvcltdAG9wZXJhdG9yIGRlbGV0ZVtdAGZpbGVzeXN0ZW0gZXJyb3I6ICVzIFsiJXMiXQBmaWxlc3lzdGVtIGVycm9yOiAlcyBbIiVzIl0gWyIlcyJdAHBpeGVsIHZlY3RvclsAc1oAX19fX1oAJWEgJWIgJWQgJUg6JU06JVMgJVkAUE9TSVgASGllcmFyY2hpY2FsTlNXAGZwVAAkVFQAJFQARW1zY3JpcHRlbkZpbGVTeXN0ZW1NYW5hZ2VyIG11c3QgYmUgaW5pdGlhbGl6ZWQgYmVmb3JlIGNhbGxpbmcgc3luY0ZTACVIOiVNOiVTAHNQAERPAHNyTgBfR0xPQkFMX19OAE5BTgAkTgBQTQBBTQBmTAAlTGFMAExDX0FMTABVYTllbmFibGVfaWZJAExBTkcASU5GAHZFAGFjdGlvbnMgJiBfVUFfQ0xFQU5VUF9QSEFTRQBhY3Rpb25zICYgX1VBX1NFQVJDSF9QSEFTRQBSRQBPRQBiMUUAYjBFAHJlc3VsdHMucmVhc29uID09IF9VUkNfSEFORExFUl9GT1VORABEQwBvcGVyYXRvcj8AZW1zY3JpcHRlbjo6bWVtb3J5X3ZpZXc8c2hvcnQ+AGVtc2NyaXB0ZW46Om1lbW9yeV92aWV3PHVuc2lnbmVkIHNob3J0PgBlbXNjcmlwdGVuOjptZW1vcnlfdmlldzxpbnQ+AGVtc2NyaXB0ZW46Om1lbW9yeV92aWV3PHVuc2lnbmVkIGludD4AZW1zY3JpcHRlbjo6bWVtb3J5X3ZpZXc8ZmxvYXQ+AGVtc2NyaXB0ZW46Om1lbW9yeV92aWV3PHVpbnQ4X3Q+AGVtc2NyaXB0ZW46Om1lbW9yeV92aWV3PGludDhfdD4AZW1zY3JpcHRlbjo6bWVtb3J5X3ZpZXc8dWludDE2X3Q+AGVtc2NyaXB0ZW46Om1lbW9yeV92aWV3PGludDE2X3Q+AGVtc2NyaXB0ZW46Om1lbW9yeV92aWV3PHVpbnQ2NF90PgBlbXNjcmlwdGVuOjptZW1vcnlfdmlldzxpbnQ2NF90PgBlbXNjcmlwdGVuOjptZW1vcnlfdmlldzx1aW50MzJfdD4AZW1zY3JpcHRlbjo6bWVtb3J5X3ZpZXc8aW50MzJfdD4Ab3BlcmF0b3I+AGVtc2NyaXB0ZW46Om1lbW9yeV92aWV3PGNoYXI+ADxjaGFyLCBzdGQ6OmNoYXJfdHJhaXRzPGNoYXI+ACwgc3RkOjphbGxvY2F0b3I8Y2hhcj4AZW1zY3JpcHRlbjo6bWVtb3J5X3ZpZXc8dW5zaWduZWQgY2hhcj4Ac3RkOjpiYXNpY19zdHJpbmc8dW5zaWduZWQgY2hhcj4AZW1zY3JpcHRlbjo6bWVtb3J5X3ZpZXc8c2lnbmVkIGNoYXI+AGVtc2NyaXB0ZW46Om1lbW9yeV92aWV3PGxvbmc+AGVtc2NyaXB0ZW46Om1lbW9yeV92aWV3PHVuc2lnbmVkIGxvbmc+AGVtc2NyaXB0ZW46Om1lbW9yeV92aWV3PGRvdWJsZT4Ab3BlcmF0b3I+PgBvcGVyYXRvcjw9PgBvcGVyYXRvci0+AG9wZXJhdG9yfD0Ab3BlcmF0b3I9AG9wZXJhdG9yXj0Ab3BlcmF0b3I+PQBvcGVyYXRvcj4+PQBvcGVyYXRvcj09AG9wZXJhdG9yPD0Ab3BlcmF0b3I8PD0Ab3BlcmF0b3IvPQBvcGVyYXRvci09AG9wZXJhdG9yKz0Ab3BlcmF0b3IqPQBvcGVyYXRvciY9AG9wZXJhdG9yJT0Ab3BlcmF0b3IhPQBvcGVyYXRvcjwAdGVtcGxhdGU8AGlkPABvcGVyYXRvcjw8AC48ACI8AFthYmk6ACBbZW5hYmxlX2lmOgBzdGQ6OgAwMTIzNDU2Nzg5AHVuc2lnbmVkIF9faW50MTI4AF9fZmxvYXQxMjgAZGVjaW1hbDEyOABDLlVURi04AGRlY2ltYWw2NABkZWNpbWFsMzIAX19neHhfcGVyc29uYWxpdHlfd2FzbTAATm90IGVub3VnaCBtZW1vcnk6IGxvYWRJbmRleCBmYWlsZWQgdG8gYWxsb2NhdGUgbGV2ZWwwAG9wZXJhdG9yLwBGYWlsZWQgdG8gcmVhZCB0aGUgaW5kZXguAEludmFsaWQgY2FsbGJhY2sgZnVuY3Rpb24gZm9yIEN1c3RvbUZpbHRlckZ1bmN0b3IuAG9wZXJhdG9yLgBJbnZhbGlkIHRoZSBmaXJzdCBhcmd1bWVudCB0eXBlLCBtdXN0IGJlIGEgbnVtYmVyLgBUaGUgbnVtYmVyIG9mIHZlY3RvcnMgYW5kIGlkcyBtdXN0IGJlIHRoZSBzYW1lLgBTZWFyY2ggaW5kZXggaGFzIG5vdCBiZWVuIGluaXRpYWxpemVkLCBjYWxsIGBpbml0SW5kZXhgIGluIGFkdmFuY2UuAENhbid0IHVzZSBhZGRQb2ludCB0byB1cGRhdGUgZGVsZXRlZCBlbGVtZW50cyBpZiByZXBsYWNlbWVudCBvZiBkZWxldGVkIGVsZW1lbnRzIGlzIGVuYWJsZWQuAFRoZSBudW1iZXIgb2YgdmVjdG9ycyBhbmQgaWRzIG11c3QgYmUgZ3JlYXRlciB0aGFuIDAuAGQuIENhbGxpbmcgc3luYyBjYWxsYmFjay4uLgBzaXplb2YuLi4AYS4gc3RhcnQgc3luY0ZTLi4uAEludmFsaWQgdGhlIG51bWJlciBvZiBrLW5lYXJlc3QgbmVpZ2hib3JzIChtdXN0IGJlIGEgcG9zaXRpdmUgbnVtYmVyKS4Ab3BlcmF0b3ItAC1pbi0Ab3BlcmF0b3ItLQBvcGVyYXRvciwAdysAb3BlcmF0b3IrAGErAG9wZXJhdG9yKysAb3BlcmF0b3IqAG9wZXJhdG9yLT4qADo6KgBvcGVyYXRvci4qAGRlY2x0eXBlKGF1dG8pAChudWxsKQAoYW5vbnltb3VzIG5hbWVzcGFjZSkAYWN0aW9ucyAmIChfVUFfU0VBUkNIX1BIQVNFIHwgX1VBX0ZPUkNFX1VOV0lORCkAYWN0aW9ucyAmIChfVUFfU0VBUkNIX1BIQVNFIHwgX1VBX0hBTkRMRVJfRlJBTUUgfCBfVUFfRk9SQ0VfVU5XSU5EKQBQYXJzZXIuRm9yd2FyZFRlbXBsYXRlUmVmcy5lbXB0eSgpACFlbXB0eSgpAG9wZXJhdG9yKCkARnJvbVBvc2l0aW9uIDw9IE5hbWVzLnNpemUoKQBTVi5zdGFydHNXaXRoKCJiYXNpY18iKQAgKAAnYmxvY2stbGl0ZXJhbCcAb3BlcmF0b3ImAG9wZXJhdG9yJiYAICYmACAmAG9wZXJhdG9yJQA+IgAoYmFzZSAhPSAwKSAmJiAiRFdfRUhfUEVfZGF0YXJlbCBpcyBpbnZhbGlkIHdpdGggYSBiYXNlIG9mIDAiAFJlcy5zdGFydHNXaXRoKCJvcGVyYXRvciIpICYmICJvcGVyYXRvciBuYW1lIGRvZXMgbm90IHN0YXJ0IHdpdGggJ29wZXJhdG9yJyIASW5kZXggPCBzaXplKCkgJiYgIkludmFsaWQgYWNjZXNzISIATGFzdCAhPSBGaXJzdCAmJiAiQ2FsbGluZyBiYWNrKCkgb24gZW1wdHkgdmVjdG9yISIATGFzdCAhPSBGaXJzdCAmJiAiUG9wcGluZyBlbXB0eSB2ZWN0b3IhIgBJbmRleCA8PSBzaXplKCkgJiYgImRyb3BCYWNrKCkgY2FuJ3QgZXhwYW5kISIAb3BlcmF0b3IhAFB1cmUgdmlydHVhbCBmdW5jdGlvbiBjYWxsZWQhAEludmFsaWQgdmVjdG9yIHNpemUgYXQgaW5kZXggAHRocm93IABub2V4Y2VwdCAALCBidXQgZ290IAAgYXQgb2Zmc2V0IABDb3VsZCBub3QgYWRkUG9pbnRzIABDb3VsZCBub3QgbWFya0RlbGV0ZUl0ZW1zIABDb3VsZCBub3QgYWRkSXRlbXMgAEludmFsaWQgdmVjdG9yIHNpemUuIE11c3QgYmUgZXF1YWwgdG8gdGhlIGRpbWVuc2lvbiBvZiB0aGUgc3BhY2UuIFRoZSBkaW1lbnNpb24gb2YgdGhlIHNwYWNlIGlzIABvcGVyYXRvciAAcmVmZXJlbmNlIHRlbXBvcmFyeSBmb3IgAHRlbXBsYXRlIHBhcmFtZXRlciBvYmplY3QgZm9yIAB0eXBlaW5mbyBmb3IgAHRocmVhZC1sb2NhbCB3cmFwcGVyIHJvdXRpbmUgZm9yIAB0aHJlYWQtbG9jYWwgaW5pdGlhbGl6YXRpb24gcm91dGluZSBmb3IgAHR5cGVpbmZvIG5hbWUgZm9yIABjb25zdHJ1Y3Rpb24gdnRhYmxlIGZvciAAZ3VhcmQgdmFyaWFibGUgZm9yIABWVFQgZm9yIABjb3ZhcmlhbnQgcmV0dXJuIHRodW5rIHRvIABub24tdmlydHVhbCB0aHVuayB0byAAaW52b2NhdGlvbiBmdW5jdGlvbiBmb3IgYmxvY2sgaW4gAGFsaWdub2YgAHNpemVvZiAAPiB0eXBlbmFtZSAAaW5pdGlhbGl6ZXIgZm9yIG1vZHVsZSAAdHlwZWlkIABJbnZhbGlkIHRoZSBnaXZlbiBhcnJheSBsZW5ndGggKGV4cGVjdGVkIAB1bnNpZ25lZCAAID8gACA9IABGYWlsZWQgdG8gcmVhZCB0aGUgaW5kZXg6IABGYWlsZWQgdG8gbm9ybWFsaXplIHRoZSBwb2ludCwgY2hlY2sgdmVjdG9yIGRpbWVuc2lvbnM6IABGYWlsZWQgdG8gY2FsbCB0aGUgY2FsbGJhY2sgZnVuY3Rpb246IABsaWJjKythYmk6IABUaGUgbWF4aW11bSBudW1iZXIgb2YgZWxlbWVudHMgaGFzIGJlZW4gcmVhY2hlZCBpbiBpbmRleCwgcGxlYXNlIGluY3JlYXNlZCB0aGUgaW5kZXggbWF4X3NpemUuICBtYXhfc2l6ZTogAFRoZSBtYXhpbXVtIG51bWJlciBvZiBlbGVtZW50cyBpbiB0aGUgaW5kZXggaGFzIGJlZW4gcmVhY2hlZC4gLCBwbGVhc2UgaW5jcmVhc2VkIHRoZSBpbmRleCBtYXhfc2l6ZS4gIG1heF9zaXplOiAAaW52YWxpZCBzcGFjZSBzaG91bGQgYmUgZXhwZWN0ZWQgbDIsIGlwLCBvciBjb3NpbmUsIG5hbWU6IABJbnZhbGlkIHRoZSBudW1iZXIgb2Ygay1uZWFyZXN0IG5laWdoYm9ycyAoY2Fubm90IGJlIGdpdmVuIGEgdmFsdWUgZ3JlYXRlciB0aGFuIGBtYXhFbGVtZW50c2A6IABITlNXTElCIEVSUk9SOiAAIDogAHNpemVvZi4uLiAAIC4uLiAALCAAb3BlcmF0b3IiIiAAVGhlIG1heGltdW0gbnVtYmVyIG9mIGVsZW1lbnRzIGhhcyBiZWVuIHJlYWNoZWQgaW4gaW5kZXgsIHBsZWFzZSBpbmNyZWFzZWQgdGhlIGluZGV4IG1heF9zaXplLiAgbWF4X3NpemU6ICV6dQoAVGhlIG1heGltdW0gbnVtYmVyIG9mIGVsZW1lbnRzIGluIHRoZSBpbmRleCBoYXMgYmVlbiByZWFjaGVkLiAsIHBsZWFzZSBpbmNyZWFzZWQgdGhlIGluZGV4IG1heF9zaXplLiAgbWF4X3NpemU6ICV6dQoAVGhlIG51bWJlciBvZiBlbGVtZW50cyBleGNlZWRzIHRoZSBzcGVjaWZpZWQgbGltaXQKAENvdWxkIG5vdCBtYXJrRGVsZXRlSXRlbXMgJXMKAENvdWxkIG5vdCBhZGRJdGVtcyAlcwoARmFpbGVkIHRvIHJlYWQgdGhlIGluZGV4OiAlcwoARmFpbGVkIHRvIG5vcm1hbGl6ZSB0aGUgcG9pbnQsIGNoZWNrIHZlY3RvciBkaW1lbnNpb25zOiAlcwoARmFpbGVkIHRvIGNhbGwgdGhlIGNhbGxiYWNrIGZ1bmN0aW9uOiAlcwoAV3JpdGVJbmRleCBmaWxlbmFtZTogJXMKAEF1dG9TYXZlIGZpbGVuYW1lOiAlcwoAaW52YWxpZCBzcGFjZSBzaG91bGQgYmUgZXhwZWN0ZWQgbDIsIGlwLCBvciBjb3NpbmUsIG5hbWU6ICVzCgBITlNXTElCIEVSUk9SOiAlcwoASW52YWxpZCB2ZWN0b3Igc2l6ZSBhdCBpbmRleCAlenUuIE11c3QgYmUgZXF1YWwgdG8gdGhlIGRpbWVuc2lvbiBvZiB0aGUgc3BhY2UuIFRoZSBkaW1lbnNpb24gb2YgdGhlIHNwYWNlIGlzICVkLgoASW52YWxpZCB2ZWN0b3Igc2l6ZS4gTXVzdCBiZSBlcXVhbCB0byB0aGUgZGltZW5zaW9uIG9mIHRoZSBzcGFjZS4gVGhlIGRpbWVuc2lvbiBvZiB0aGUgc3BhY2UgaXMgJWQuCgBJbnZhbGlkIHRoZSBnaXZlbiBhcnJheSBsZW5ndGggKGV4cGVjdGVkICVsdSwgYnV0IGdvdCAlenUpLgoASW52YWxpZCB0aGUgbnVtYmVyIG9mIGstbmVhcmVzdCBuZWlnaGJvcnMgKGNhbm5vdCBiZSBnaXZlbiBhIHZhbHVlIGdyZWF0ZXIgdGhhbiBgbWF4RWxlbWVudHNgOiAlenUpLgoAAOiEAAAYKgAATjEwZW1zY3JpcHRlbjN2YWxFAACshQAABCoAAE4xMGVtc2NyaXB0ZW4xMW1lbW9yeV92aWV3SWZFRQAArIUAACAqAAAYKgAAGCoAAGlpaQBOMTBlbXNjcmlwdGVuN0wyU3BhY2VFAACshQAAVCoAAFBOMTBlbXNjcmlwdGVuN0wyU3BhY2VFAIyGAAB0KgAAAAAAAGwqAABQS04xMGVtc2NyaXB0ZW43TDJTcGFjZUUAAAAAjIYAAJwqAAABAAAAbCoAAGlpAHYAdmkAjCoAAFSFAAAAAAAAMCsAAG0AAABuAAAAbwAAAHAAAABxAAAATjdobnN3bGliN0wyU3BhY2VFAE43aG5zd2xpYjE0U3BhY2VJbnRlcmZhY2VJZkVFAAAAAKyFAAAHKwAA1IUAAPQqAAAoKwAAAAAAAJCFAACMKgAAGCoAABgqAABmaWlpaQAAAFSFAACMKgAATjEwZW1zY3JpcHRlbjE3SW5uZXJQcm9kdWN0U3BhY2VFAAAArIUAAGArAABQTjEwZW1zY3JpcHRlbjE3SW5uZXJQcm9kdWN0U3BhY2VFAACMhgAAjCsAAAAAAACEKwAAUEtOMTBlbXNjcmlwdGVuMTdJbm5lclByb2R1Y3RTcGFjZUUAjIYAAMArAAABAAAAhCsAALArAABUhQAAAAAAADgsAAByAAAAcwAAAHQAAABwAAAAdQAAAE43aG5zd2xpYjE3SW5uZXJQcm9kdWN0U3BhY2VFAAAA1IUAABgsAAAoKwBB0NgAC7UFkIUAALArAAAYKgAAGCoAAFSFAACwKwAAAAAAAMAsAAB2AAAATjEwZW1zY3JpcHRlbjE5Q3VzdG9tRmlsdGVyRnVuY3RvckUATjdobnN3bGliMTdCYXNlRmlsdGVyRnVuY3RvckUAAACshQAAmCwAANSFAAB0LAAAuCwAAACFAAAYKgAAbIUAAFBOMTBlbXNjcmlwdGVuMTlDdXN0b21GaWx0ZXJGdW5jdG9yRQAAAACMhgAA2CwAAAAAAADALAAAUEtOMTBlbXNjcmlwdGVuMTlDdXN0b21GaWx0ZXJGdW5jdG9yRQAAAIyGAAAQLQAAAQAAAMAsAAAALQAAGCoAAACFAAAALQAAbIUAAGlpaWkATjEwZW1zY3JpcHRlbjE2QnJ1dGVmb3JjZVNlYXJjaEUAAACshQAAYS0AAFBOMTBlbXNjcmlwdGVuMTZCcnV0ZWZvcmNlU2VhcmNoRQAAAIyGAACMLQAAAAAAAIQtAABQS04xMGVtc2NyaXB0ZW4xNkJydXRlZm9yY2VTZWFyY2hFAACMhgAAwC0AAAEAAACELQAAsC0AAEAuAABUhQAATlN0M19fMjEyYmFzaWNfc3RyaW5nSWNOU18xMWNoYXJfdHJhaXRzSWNFRU5TXzlhbGxvY2F0b3JJY0VFRUUAAKyFAAAALgAA6IQAALAtAABUhQAAdmlpaQAAAAAAAAAAyC4AAHcAAAB4AAAAeQAAAHoAAAB7AAAAfAAAAE43aG5zd2xpYjE2QnJ1dGVmb3JjZVNlYXJjaElmRUUATjdobnN3bGliMThBbGdvcml0aG1JbnRlcmZhY2VJZkVFAAAArIUAAJwuAADUhQAAfC4AAMAuAAAAAAAAGCoAALAtAADohAAAsC0AAEAuAAAAAAAA6IQAALAtAAAYKgAAVIUAAHZpaWlpAEGQ3gALsgEYKgAAsC0AABgqAABUhQAAGCoAAGlpaWlpaQAAVIUAALAtAABOMTBlbXNjcmlwdGVuMTVIaWVyYXJjaGljYWxOU1dFAKyFAAA0LwAAUE4xMGVtc2NyaXB0ZW4xNUhpZXJhcmNoaWNhbE5TV0UAAAAAjIYAAFwvAAAAAAAAVC8AAFBLTjEwZW1zY3JpcHRlbjE1SGllcmFyY2hpY2FsTlNXRQAAAIyGAACQLwAAAQAAAFQvAEHQ3wALFYAvAABALgAAVIUAAEAuAABpaWlpaQBB8N8AC3LohAAAgC8AAFSFAABUhQAAVIUAAFSFAAB2aWlpaWlpAAAAAABQMAAAfQAAAH4AAAB5AAAAfwAAAIAAAACBAAAATjdobnN3bGliMTVIaWVyYXJjaGljYWxOU1dJZkVFAADUhQAAMDAAAMAuAAAYKgAAgC8AQfDgAAsy6IQAAIAvAABALgAAVIUAAOiEAACALwAAQC4AAOiEAACALwAAVIUAABgqAACALwAAVIUAQbDhAAuSAeiEAACALwAAGCoAAFSFAAAAhQAAdmlpaWlpAAAAAAAA6IQAAIAvAAAYKgAAGCoAAACFAABOMTBlbXNjcmlwdGVuMTFtZW1vcnlfdmlld0lqRUUAAKyFAADkMAAAAAAAABgqAACALwAAGCoAAACFAAAYKgAAgC8AAEiFAACALwAA6IQAAIAvAAAYKgAAVIUAALQvAEHQ4gALsSkYKgAAgC8AABgqAABUhQAAGCoAAOiEAAAAhQAAdmlpAE4xMGVtc2NyaXB0ZW4yN0Vtc2NyaXB0ZW5GaWxlU3lzdGVtTWFuYWdlckUArIUAAHAxAABQTjEwZW1zY3JpcHRlbjI3RW1zY3JpcHRlbkZpbGVTeXN0ZW1NYW5hZ2VyRQAAAACMhgAApDEAAAAAAACcMQAAUEtOMTBlbXNjcmlwdGVuMjdFbXNjcmlwdGVuRmlsZVN5c3RlbU1hbmFnZXJFAAAAjIYAAOQxAAABAAAAnDEAANQxAADohAAAQC4AAGlpAAAAhQAA6IQAAACFAAAYKgAAAIUAAEAuAABOU3QzX18yMTJiYXNpY19zdHJpbmdJaE5TXzExY2hhcl90cmFpdHNJaEVFTlNfOWFsbG9jYXRvckloRUVFRQAArIUAAEwyAABOU3QzX18yMTJiYXNpY19zdHJpbmdJd05TXzExY2hhcl90cmFpdHNJd0VFTlNfOWFsbG9jYXRvckl3RUVFRQAArIUAAJQyAABOU3QzX18yMTJiYXNpY19zdHJpbmdJRHNOU18xMWNoYXJfdHJhaXRzSURzRUVOU185YWxsb2NhdG9ySURzRUVFRQAAAKyFAADcMgAATlN0M19fMjEyYmFzaWNfc3RyaW5nSURpTlNfMTFjaGFyX3RyYWl0c0lEaUVFTlNfOWFsbG9jYXRvcklEaUVFRUUAAACshQAAKDMAAE4xMGVtc2NyaXB0ZW4xMW1lbW9yeV92aWV3SWNFRQAArIUAAHQzAABOMTBlbXNjcmlwdGVuMTFtZW1vcnlfdmlld0lhRUUA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YXNlN2ZhaWx1cmVFAAAA1IUAANxMAACsggAATlN0M19fMjE5X19pb3N0cmVhbV9jYXRlZ29yeUUAAADUhQAABE0AAPSCAADRdJ4AV529KoBwUg///z4nCgAAAGQAAADoAwAAECcAAKCGAQBAQg8AgJaYAADh9QUYAAAANQAAAHEAAABr////zvv//5K///8AAAAAAAAAAP////////////////////////////////////////////////////////////////8AAQIDBAUGBwgJ/////////woLDA0ODxAREhMUFRYXGBkaGxwdHh8gISIj////////CgsMDQ4PEBESExQVFhcYGRobHB0eHyAhIiP/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////AAECBAcDBgUAAAAAAAAAAgAAwAMAAMAEAADABQAAwAYAAMAHAADACAAAwAkAAMAKAADACwAAwAwAAMANAADADgAAwA8AAMAQAADAEQAAwBIAAMATAADAFAAAwBUAAMAWAADAFwAAwBgAAMAZAADAGgAAwBsAAMAcAADAHQAAwB4AAMAfAADAAAAAswEAAMMCAADDAwAAwwQAAMMFAADDBgAAwwcAAMMIAADDCQAAwwoAAMMLAADDDAAAww0AANMOAADDDwAAwwAADLsBAAzDAgAMwwMADMMEAAzbAAAAAN4SBJUAAAAA////////////////YE8AABQAAABDLlVURi04AEGwnwELAnRPAEHQnwELSkxDX0NUWVBFAAAAAExDX05VTUVSSUMAAExDX1RJTUUAAAAAAExDX0NPTExBVEUAAExDX01PTkVUQVJZAExDX01FU1NBR0VTACBSAEGkpAEL+QMBAAAAAgAAAAMAAAAEAAAABQAAAAYAAAAHAAAACAAAAAkAAAAKAAAACwAAAAwAAAANAAAADgAAAA8AAAAQAAAAEQAAABIAAAATAAAAFAAAABUAAAAWAAAAFwAAABgAAAAZAAAAGgAAABsAAAAcAAAAHQAAAB4AAAAfAAAAIAAAACEAAAAiAAAA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AQAAfwEAAIABAACBAQAAggEAAIMBAACEAQAAAAAAAEB3AADHAQAAyAEAAMkBAABOU3QzX18yMTRfX3NoYXJlZF9jb3VudEUAAAAArIUAACR3AAAAAAAAhHcAAMcBAADKAQAAyQEAAMsBAADJAQAATlN0M19fMjE5X19zaGFyZWRfd2Vha19jb3VudEUAAAAwhgAAZHcAAAAAAAABAAAAQHcAQaDvAQvZENx3AADMAQAAzQEAAM4BAABOU3QzX18yNF9fZnMxMGZpbGVzeXN0ZW0xNmZpbGVzeXN0ZW1fZXJyb3JFANSFAACwdwAArIIAAAAAAABoeAAAzwEAANABAADRAQAAywEAANIBAABOU3QzX18yMjBfX3NoYXJlZF9wdHJfZW1wbGFjZUlOU180X19mczEwZmlsZXN5c3RlbTE2ZmlsZXN5c3RlbV9lcnJvcjhfU3RvcmFnZUVOU185YWxsb2NhdG9ySVM0X0VFRUUA1IUAAAR4AACEdwAABgUIAggECAEIAwgHTm8gZXJyb3IgaW5mb3JtYXRpb24ASWxsZWdhbCBieXRlIHNlcXVlbmNlAERvbWFpbiBlcnJvcgBSZXN1bHQgbm90IHJlcHJlc2VudGFibGUATm90IGEgdHR5AFBlcm1pc3Npb24gZGVuaWVkAE9wZXJhdGlvbiBub3QgcGVybWl0dGVkAE5vIHN1Y2ggZmlsZSBvciBkaXJlY3RvcnkATm8gc3VjaCBwcm9jZXNzAEZpbGUgZXhpc3RzAFZhbHVlIHRvbyBsYXJnZSBmb3IgZGF0YSB0eXBlAE5vIHNwYWNlIGxlZnQgb24gZGV2aWNlAE91dCBvZiBtZW1vcnkAUmVzb3VyY2UgYnVzeQBJbnRlcnJ1cHRlZCBzeXN0ZW0gY2FsbABSZXNvdXJjZSB0ZW1wb3JhcmlseSB1bmF2YWlsYWJsZQBJbnZhbGlkIHNlZWsAQ3Jvc3MtZGV2aWNlIGxpbmsAUmVhZC1vbmx5IGZpbGUgc3lzdGVtAERpcmVjdG9yeSBub3QgZW1wdHkAQ29ubmVjdGlvbiByZXNldCBieSBwZWVyAE9wZXJhdGlvbiB0aW1lZCBvdXQAQ29ubmVjdGlvbiByZWZ1c2VkAEhvc3QgaXMgZG93bgBIb3N0IGlzIHVucmVhY2hhYmxlAEFkZHJlc3MgaW4gdXNlAEJyb2tlbiBwaXBlAEkvTyBlcnJvcgBObyBzdWNoIGRldmljZSBvciBhZGRyZXNzAEJsb2NrIGRldmljZSByZXF1aXJlZABObyBzdWNoIGRldmljZQBOb3QgYSBkaXJlY3RvcnkASXMgYSBkaXJlY3RvcnkAVGV4dCBmaWxlIGJ1c3kARXhlYyBmb3JtYXQgZXJyb3IASW52YWxpZCBhcmd1bWVudABBcmd1bWVudCBsaXN0IHRvbyBsb25nAFN5bWJvbGljIGxpbmsgbG9vcABGaWxlbmFtZSB0b28gbG9uZwBUb28gbWFueSBvcGVuIGZpbGVzIGluIHN5c3RlbQBObyBmaWxlIGRlc2NyaXB0b3JzIGF2YWlsYWJsZQBCYWQgZmlsZSBkZXNjcmlwdG9yAE5vIGNoaWxkIHByb2Nlc3MAQmFkIGFkZHJlc3MARmlsZSB0b28gbGFyZ2UAVG9vIG1hbnkgbGlua3MATm8gbG9ja3MgYXZhaWxhYmxlAFJlc291cmNlIGRlYWRsb2NrIHdvdWxkIG9jY3VyAFN0YXRlIG5vdCByZWNvdmVyYWJsZQBQcmV2aW91cyBvd25lciBkaWVkAE9wZXJhdGlvbiBjYW5jZWxlZABGdW5jdGlvbiBub3QgaW1wbGVtZW50ZWQATm8gbWVzc2FnZSBvZiBkZXNpcmVkIHR5cGUASWRlbnRpZmllciByZW1vdmVkAERldmljZSBub3QgYSBzdHJlYW0ATm8gZGF0YSBhdmFpbGFibGUARGV2aWNlIHRpbWVvdXQAT3V0IG9mIHN0cmVhbXMgcmVzb3VyY2VzAExpbmsgaGFzIGJlZW4gc2V2ZXJlZABQcm90b2NvbCBlcnJvcgBCYWQgbWVzc2FnZQBGaWxlIGRlc2NyaXB0b3IgaW4gYmFkIHN0YXRlAE5vdCBhIHNvY2tldABEZXN0aW5hdGlvbiBhZGRyZXNzIHJlcXVpcmVkAE1lc3NhZ2UgdG9vIGxhcmdlAFByb3RvY29sIHdyb25nIHR5cGUgZm9yIHNvY2tldABQcm90b2NvbCBub3QgYXZhaWxhYmxlAFByb3RvY29sIG5vdCBzdXBwb3J0ZWQAU29ja2V0IHR5cGUgbm90IHN1cHBvcnRlZABOb3Qgc3VwcG9ydGVkAFByb3RvY29sIGZhbWlseSBu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b2ludGVyVHlwZUUA1IUAAOirAACciQAAAAAAAISsAADMAgAABAIAAAUCAAAGAgAAzQIAAM4CAAAJAgAACgIAAM8CAABOMTJfR0xPQkFMX19OXzExNml0YW5pdW1fZGVtYW5nbGUxM1JlZmVyZW5jZVR5cGVFAAAA1IUAAFCsAACciQAAAAAAAPisAAADAgAABAIAAAUCAAAGAgAA0AIAAAgCAAAJAgAACgIAANECAABOMTJfR0xPQkFMX19OXzExNml0YW5pdW1fZGVtYW5nbGUyMFBvc3RmaXhRdWFsaWZpZWRUeXBlRQAAAADUhQAAvKwAAJyJAABpCwAAiREAAIkRAACNDgAAfw4AAHAOAEGg2gILAQUAQazaAgsBgwBBxNoCCw6EAAAAhQAAADiyAAAABABB3NoCCwEBAEHs2gILBf////8KAEGw2wILGSCtAABwwwEAgEwAAECCAABkggAAAAAAAAUAQdTbAgsBigBB7NsCCwqEAAAAiAAAAEzBAEGE3AILAQIAQZTcAgsI//////////8AQdjcAgsOyK0AAN4BAADfAQAALggA3bAIBG5hbWUBgasI+AcADV9lbXZhbF9kZWNyZWYBGF9lbXZhbF9nZXRfbWV0aG9kX2NhbGxlcgINX2VtdmFsX2luY3JlZgMXX2VtdmFsX2NhbGxfdm9pZF9tZXRob2QEGV9lbWJpbmRfcmVnaXN0ZXJfZnVuY3Rpb24FFl9lbWJpbmRfcmVnaXN0ZXJfY2xhc3MGIl9lbWJpbmRfcmVnaXN0ZXJfY2xhc3NfY29uc3RydWN0b3IHH19lbWJpbmRfcmVnaXN0ZXJfY2xhc3NfZnVuY3Rpb24IJV9lbWJpbmRfcmVnaXN0ZXJfY2xhc3NfY2xhc3NfZnVuY3Rpb24JEV9lbXZhbF90YWtlX3ZhbHVlChBfZW12YWxfbmV3X2FycmF5CxFfZW12YWxfbmV3X29iamVjdAwSX2VtdmFsX25ld19jc3RyaW5nDRNfZW12YWxfc2V0X3Byb3BlcnR5DhhlbXNjcmlwdGVuX2FzbV9jb25zdF9pbnQPCnN5bmNJZGJfanMQIV9lbXZhbF9uZXdfYXJyYXlfZnJvbV9tZW1vcnlfdmlldxETX2VtdmFsX2dldF9wcm9wZXJ0eRIJX2VtdmFsX2FzExZfZW12YWxfcnVuX2Rlc3RydWN0b3JzFBJfZW12YWxfY2FsbF9tZXRob2QVDV9fYXNzZXJ0X2ZhaWwWFV9lbWJpbmRfcmVnaXN0ZXJfdm9pZBcVX2VtYmluZF9yZWdpc3Rlcl9ib29sGBhfZW1iaW5kX3JlZ2lzdGVyX2ludGVnZXIZFl9lbWJpbmRfcmVnaXN0ZXJfZmxvYXQaG19lbWJpbmRfcmVnaXN0ZXJfc3RkX3N0cmluZxscX2VtYmluZF9yZWdpc3Rlcl9zdGRfd3N0cmluZxwWX2VtYmluZF9yZWdpc3Rlcl9lbXZhbB0cX2VtYmluZF9yZWdpc3Rlcl9tZW1vcnlfdmlldx4VZW1zY3JpcHRlbl9tZW1jcHlfYmlnHw9fX3dhc2lfZmRfd3JpdGUgFmVtc2NyaXB0ZW5fcmVzaXplX2hlYXAhEF9fc3lzY2FsbF9vcGVuYXQiEV9fc3lzY2FsbF9mY250bDY0Iw9fX3N5c2NhbGxfaW9jdGwkDl9fd2FzaV9mZF9yZWFkJQ9fX3dhc2lfZmRfY2xvc2UmGF9fd2FzaV9lbnZpcm9uX3NpemVzX2dldCcSX193YXNpX2Vudmlyb25fZ2V0KApzdHJmdGltZV9sKRBfX3N5c2NhbGxfc3RhdDY0KgVhYm9ydCsiX190aHJvd19leGNlcHRpb25fd2l0aF9zdGFja190cmFjZSwjbGVnYWxpbXBvcnQkX2VtYmluZF9yZWdpc3Rlcl9iaWdpbnQtGmxlZ2FsaW1wb3J0JF9fd2FzaV9mZF9zZWVrLhFfX3dhc21fY2FsbF9jdG9ycy8XX19jeHhfZ2xvYmFsX2FycmF5X2R0b3IwGV9fY3h4X2dsb2JhbF9hcnJheV9kdG9yLjQxGV9fY3h4X2dsb2JhbF9hcnJheV9kdG9yLjYyiwFzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPiwgc3RkOjpfXzI6OmFsbG9jYXRvcjxjaGFyPj46OmJhc2ljX3N0cmluZ1thYmk6djE1MDA3XTxzdGQ6Om51bGxwdHJfdD4oY2hhciBjb25zdCopM35zdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPiwgc3RkOjpfXzI6OmFsbG9jYXRvcjxjaGFyPj46Ol9fdGhyb3dfbGVuZ3RoX2Vycm9yW2FiaTp2MTUwMDddKCkgY29uc3Q0GV9fY3h4X2dsb2JhbF9hcnJheV9kdG9yLjg1DnNldElkYmZzU3luY2VkNltlbXNjcmlwdGVuOjpub3JtYWxpemVQb2ludHNQdXJlKHN0ZDo6X18yOjp2ZWN0b3I8ZmxvYXQsIHN0ZDo6X18yOjphbGxvY2F0b3I8ZmxvYXQ+PiBjb25zdCYpN11zdGQ6Ol9fMjo6dmVjdG9yPGZsb2F0LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPGZsb2F0Pj46Ol9fdGhyb3dfbGVuZ3RoX2Vycm9yW2FiaTp2MTUwMDddKCkgY29uc3Q4IWVtc2NyaXB0ZW46OmVtYmluZF9pbml0X2huc3dsaWIoKTmhAmVtc2NyaXB0ZW46OmludGVybmFsOjpJbnZva2VyPHN0ZDo6X18yOjp2ZWN0b3I8ZmxvYXQsIHN0ZDo6X18yOjphbGxvY2F0b3I8ZmxvYXQ+Piwgc3RkOjpfXzI6OnZlY3RvcjxmbG9hdCwgc3RkOjpfXzI6OmFsbG9jYXRvcjxmbG9hdD4+IGNvbnN0Jj46Omludm9rZShzdGQ6Ol9fMjo6dmVjdG9yPGZsb2F0LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPGZsb2F0Pj4gKCopKHN0ZDo6X18yOjp2ZWN0b3I8ZmxvYXQsIHN0ZDo6X18yOjphbGxvY2F0b3I8ZmxvYXQ+PiBjb25zdCYpLCBlbXNjcmlwdGVuOjpfRU1fVkFMKik6WnZvaWQgY29uc3QqIGVtc2NyaXB0ZW46OmludGVybmFsOjpnZXRBY3R1YWxUeXBlPGVtc2NyaXB0ZW46OkwyU3BhY2U+KGVtc2NyaXB0ZW46OkwyU3BhY2UqKTtUdm9pZCBlbXNjcmlwdGVuOjppbnRlcm5hbDo6cmF3X2Rlc3RydWN0b3I8ZW1zY3JpcHRlbjo6TDJTcGFjZT4oZW1zY3JpcHRlbjo6TDJTcGFjZSopPIMBZW1zY3JpcHRlbjo6aW50ZXJuYWw6Okludm9rZXI8ZW1zY3JpcHRlbjo6TDJTcGFjZSosIHVuc2lnbmVkIGludCYmPjo6aW52b2tlKGVtc2NyaXB0ZW46OkwyU3BhY2UqICgqKSh1bnNpZ25lZCBpbnQmJiksIHVuc2lnbmVkIGludCk9amVtc2NyaXB0ZW46OkwyU3BhY2UqIGVtc2NyaXB0ZW46OmludGVybmFsOjpvcGVyYXRvcl9uZXc8ZW1zY3JpcHRlbjo6TDJTcGFjZSwgdW5zaWduZWQgaW50Pih1bnNpZ25lZCBpbnQmJik+lQFlbXNjcmlwdGVuOjpMMlNwYWNlOjpkaXN0YW5jZShzdGQ6Ol9fMjo6dmVjdG9yPGZsb2F0LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPGZsb2F0Pj4gY29uc3QmLCBzdGQ6Ol9fMjo6dmVjdG9yPGZsb2F0LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPGZsb2F0Pj4gY29uc3QmKT+5BGVtc2NyaXB0ZW46OmludGVybmFsOjpNZXRob2RJbnZva2VyPGZsb2F0IChlbXNjcmlwdGVuOjpMMlNwYWNlOjoqKShzdGQ6Ol9fMjo6dmVjdG9yPGZsb2F0LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPGZsb2F0Pj4gY29uc3QmLCBzdGQ6Ol9fMjo6dmVjdG9yPGZsb2F0LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPGZsb2F0Pj4gY29uc3QmKSwgZmxvYXQsIGVtc2NyaXB0ZW46OkwyU3BhY2UqLCBzdGQ6Ol9fMjo6dmVjdG9yPGZsb2F0LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPGZsb2F0Pj4gY29uc3QmLCBzdGQ6Ol9fMjo6dmVjdG9yPGZsb2F0LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPGZsb2F0Pj4gY29uc3QmPjo6aW52b2tlKGZsb2F0IChlbXNjcmlwdGVuOjpMMlNwYWNlOjoqIGNvbnN0Jikoc3RkOjpfXzI6OnZlY3RvcjxmbG9hdCwgc3RkOjpfXzI6OmFsbG9jYXRvcjxmbG9hdD4+IGNvbnN0Jiwgc3RkOjpfXzI6OnZlY3RvcjxmbG9hdCwgc3RkOjpfXzI6OmFsbG9jYXRvcjxmbG9hdD4+IGNvbnN0JiksIGVtc2NyaXB0ZW46OkwyU3BhY2UqLCBlbXNjcmlwdGVuOjpfRU1fVkFMKiwgZW1zY3JpcHRlbjo6X0VNX1ZBTCopQCdlbXNjcmlwdGVuOjpMMlNwYWNlOjpnZXROdW1EaW1lbnNpb25zKClBvgFlbXNjcmlwdGVuOjppbnRlcm5hbDo6TWV0aG9kSW52b2tlcjx1bnNpZ25lZCBpbnQgKGVtc2NyaXB0ZW46OkwyU3BhY2U6OiopKCksIHVuc2lnbmVkIGludCwgZW1zY3JpcHRlbjo6TDJTcGFjZSo+OjppbnZva2UodW5zaWduZWQgaW50IChlbXNjcmlwdGVuOjpMMlNwYWNlOjoqIGNvbnN0JikoKSwgZW1zY3JpcHRlbjo6TDJTcGFjZSopQm52b2lkIGNvbnN0KiBlbXNjcmlwdGVuOjppbnRlcm5hbDo6Z2V0QWN0dWFsVHlwZTxlbXNjcmlwdGVuOjpJbm5lclByb2R1Y3RTcGFjZT4oZW1zY3JpcHRlbjo6SW5uZXJQcm9kdWN0U3BhY2UqKUN+ZW1zY3JpcHRlbjo6SW5uZXJQcm9kdWN0U3BhY2UqIGVtc2NyaXB0ZW46OmludGVybmFsOjpvcGVyYXRvcl9uZXc8ZW1zY3JpcHRlbjo6SW5uZXJQcm9kdWN0U3BhY2UsIHVuc2lnbmVkIGludD4odW5zaWduZWQgaW50JiYpRJ8BZW1zY3JpcHRlbjo6SW5uZXJQcm9kdWN0U3BhY2U6OmRpc3RhbmNlKHN0ZDo6X18yOjp2ZWN0b3I8ZmxvYXQsIHN0ZDo6X18yOjphbGxvY2F0b3I8ZmxvYXQ+PiBjb25zdCYsIHN0ZDo6X18yOjp2ZWN0b3I8ZmxvYXQsIHN0ZDo6X18yOjphbGxvY2F0b3I8ZmxvYXQ+PiBjb25zdCYpRXJ2b2lkIGNvbnN0KiBlbXNjcmlwdGVuOjppbnRlcm5hbDo6Z2V0QWN0dWFsVHlwZTxlbXNjcmlwdGVuOjpDdXN0b21GaWx0ZXJGdW5jdG9yPihlbXNjcmlwdGVuOjpDdXN0b21GaWx0ZXJGdW5jdG9yKilGbHZvaWQgZW1zY3JpcHRlbjo6aW50ZXJuYWw6OnJhd19kZXN0cnVjdG9yPGVtc2NyaXB0ZW46OkN1c3RvbUZpbHRlckZ1bmN0b3I+KGVtc2NyaXB0ZW46OkN1c3RvbUZpbHRlckZ1bmN0b3IqKUepAWVtc2NyaXB0ZW46OmludGVybmFsOjpJbnZva2VyPGVtc2NyaXB0ZW46OkN1c3RvbUZpbHRlckZ1bmN0b3IqLCBlbXNjcmlwdGVuOjp2YWwmJj46Omludm9rZShlbXNjcmlwdGVuOjpDdXN0b21GaWx0ZXJGdW5jdG9yKiAoKikoZW1zY3JpcHRlbjo6dmFsJiYpLCBlbXNjcmlwdGVuOjpfRU1fVkFMKilIiAFlbXNjcmlwdGVuOjpDdXN0b21GaWx0ZXJGdW5jdG9yKiBlbXNjcmlwdGVuOjppbnRlcm5hbDo6b3BlcmF0b3JfbmV3PGVtc2NyaXB0ZW46OkN1c3RvbUZpbHRlckZ1bmN0b3IsIGVtc2NyaXB0ZW46OnZhbD4oZW1zY3JpcHRlbjo6dmFsJiYpSY4CZW1zY3JpcHRlbjo6aW50ZXJuYWw6Ok1ldGhvZEludm9rZXI8Ym9vbCAoZW1zY3JpcHRlbjo6Q3VzdG9tRmlsdGVyRnVuY3Rvcjo6KikodW5zaWduZWQgbG9uZyksIGJvb2wsIGVtc2NyaXB0ZW46OkN1c3RvbUZpbHRlckZ1bmN0b3IqLCB1bnNpZ25lZCBsb25nPjo6aW52b2tlKGJvb2wgKGVtc2NyaXB0ZW46OkN1c3RvbUZpbHRlckZ1bmN0b3I6OiogY29uc3QmKSh1bnNpZ25lZCBsb25nKSwgZW1zY3JpcHRlbjo6Q3VzdG9tRmlsdGVyRnVuY3RvciosIHVuc2lnbmVkIGxvbmcpSmx2b2lkIGNvbnN0KiBlbXNjcmlwdGVuOjppbnRlcm5hbDo6Z2V0QWN0dWFsVHlwZTxlbXNjcmlwdGVuOjpCcnV0ZWZvcmNlU2VhcmNoPihlbXNjcmlwdGVuOjpCcnV0ZWZvcmNlU2VhcmNoKilLZnZvaWQgZW1zY3JpcHRlbjo6aW50ZXJuYWw6OnJhd19kZXN0cnVjdG9yPGVtc2NyaXB0ZW46OkJydXRlZm9yY2VTZWFyY2g+KGVtc2NyaXB0ZW46OkJydXRlZm9yY2VTZWFyY2gqKUzQA2Vtc2NyaXB0ZW46OmludGVybmFsOjpJbnZva2VyPGVtc2NyaXB0ZW46OkJydXRlZm9yY2VTZWFyY2gqLCBzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPiwgc3RkOjpfXzI6OmFsbG9jYXRvcjxjaGFyPj4mJiwgdW5zaWduZWQgaW50JiY+OjppbnZva2UoZW1zY3JpcHRlbjo6QnJ1dGVmb3JjZVNlYXJjaCogKCopKHN0ZDo6X18yOjpiYXNpY19zdHJpbmc8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPGNoYXI+PiYmLCB1bnNpZ25lZCBpbnQmJiksIGVtc2NyaXB0ZW46OmludGVybmFsOjpCaW5kaW5nVHlwZTxzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPiwgc3RkOjpfXzI6OmFsbG9jYXRvcjxjaGFyPj4sIHZvaWQ+OjondW5uYW1lZCcqLCB1bnNpZ25lZCBpbnQpTaoCZW1zY3JpcHRlbjo6QnJ1dGVmb3JjZVNlYXJjaCogZW1zY3JpcHRlbjo6aW50ZXJuYWw6Om9wZXJhdG9yX25ldzxlbXNjcmlwdGVuOjpCcnV0ZWZvcmNlU2VhcmNoLCBzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPiwgc3RkOjpfXzI6OmFsbG9jYXRvcjxjaGFyPj4sIHVuc2lnbmVkIGludD4oc3RkOjpfXzI6OmJhc2ljX3N0cmluZzxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4sIHN0ZDo6X18yOjphbGxvY2F0b3I8Y2hhcj4+JiYsIHVuc2lnbmVkIGludCYmKU41ZW1zY3JpcHRlbjo6QnJ1dGVmb3JjZVNlYXJjaDo6aW5pdEluZGV4KHVuc2lnbmVkIGludClP/gFlbXNjcmlwdGVuOjppbnRlcm5hbDo6TWV0aG9kSW52b2tlcjx2b2lkIChlbXNjcmlwdGVuOjpCcnV0ZWZvcmNlU2VhcmNoOjoqKSh1bnNpZ25lZCBpbnQpLCB2b2lkLCBlbXNjcmlwdGVuOjpCcnV0ZWZvcmNlU2VhcmNoKiwgdW5zaWduZWQgaW50Pjo6aW52b2tlKHZvaWQgKGVtc2NyaXB0ZW46OkJydXRlZm9yY2VTZWFyY2g6OiogY29uc3QmKSh1bnNpZ25lZCBpbnQpLCBlbXNjcmlwdGVuOjpCcnV0ZWZvcmNlU2VhcmNoKiwgdW5zaWduZWQgaW50KVAyZW1zY3JpcHRlbjo6QnJ1dGVmb3JjZVNlYXJjaDo6aXNJbmRleEluaXRpYWxpemVkKClR6wFlbXNjcmlwdGVuOjppbnRlcm5hbDo6TWV0aG9kSW52b2tlcjxlbXNjcmlwdGVuOjp2YWwgKGVtc2NyaXB0ZW46OkJydXRlZm9yY2VTZWFyY2g6OiopKCksIGVtc2NyaXB0ZW46OnZhbCwgZW1zY3JpcHRlbjo6QnJ1dGVmb3JjZVNlYXJjaCo+OjppbnZva2UoZW1zY3JpcHRlbjo6dmFsIChlbXNjcmlwdGVuOjpCcnV0ZWZvcmNlU2VhcmNoOjoqIGNvbnN0JikoKSwgZW1zY3JpcHRlbjo6QnJ1dGVmb3JjZVNlYXJjaCopUoQBZW1zY3JpcHRlbjo6QnJ1dGVmb3JjZVNlYXJjaDo6cmVhZEluZGV4KHN0ZDo6X18yOjpiYXNpY19zdHJpbmc8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPGNoYXI+PiBjb25zdCYpU+gEZW1zY3JpcHRlbjo6aW50ZXJuYWw6Ok1ldGhvZEludm9rZXI8dm9pZCAoZW1zY3JpcHRlbjo6QnJ1dGVmb3JjZVNlYXJjaDo6Kikoc3RkOjpfXzI6OmJhc2ljX3N0cmluZzxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4sIHN0ZDo6X18yOjphbGxvY2F0b3I8Y2hhcj4+IGNvbnN0JiksIHZvaWQsIGVtc2NyaXB0ZW46OkJydXRlZm9yY2VTZWFyY2gqLCBzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPiwgc3RkOjpfXzI6OmFsbG9jYXRvcjxjaGFyPj4gY29uc3QmPjo6aW52b2tlKHZvaWQgKGVtc2NyaXB0ZW46OkJydXRlZm9yY2VTZWFyY2g6OiogY29uc3QmKShzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPiwgc3RkOjpfXzI6OmFsbG9jYXRvcjxjaGFyPj4gY29uc3QmKSwgZW1zY3JpcHRlbjo6QnJ1dGVmb3JjZVNlYXJjaCosIGVtc2NyaXB0ZW46OmludGVybmFsOjpCaW5kaW5nVHlwZTxzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPiwgc3RkOjpfXzI6OmFsbG9jYXRvcjxjaGFyPj4sIHZvaWQ+OjondW5uYW1lZCcqKVSFAWVtc2NyaXB0ZW46OkJydXRlZm9yY2VTZWFyY2g6OndyaXRlSW5kZXgoc3RkOjpfXzI6OmJhc2ljX3N0cmluZzxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4sIHN0ZDo6X18yOjphbGxvY2F0b3I8Y2hhcj4+IGNvbnN0JilVcGVtc2NyaXB0ZW46OkJydXRlZm9yY2VTZWFyY2g6OmFkZFBvaW50KHN0ZDo6X18yOjp2ZWN0b3I8ZmxvYXQsIHN0ZDo6X18yOjphbGxvY2F0b3I8ZmxvYXQ+PiBjb25zdCYsIHVuc2lnbmVkIGludClWyANlbXNjcmlwdGVuOjppbnRlcm5hbDo6TWV0aG9kSW52b2tlcjx2b2lkIChlbXNjcmlwdGVuOjpCcnV0ZWZvcmNlU2VhcmNoOjoqKShzdGQ6Ol9fMjo6dmVjdG9yPGZsb2F0LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPGZsb2F0Pj4gY29uc3QmLCB1bnNpZ25lZCBpbnQpLCB2b2lkLCBlbXNjcmlwdGVuOjpCcnV0ZWZvcmNlU2VhcmNoKiwgc3RkOjpfXzI6OnZlY3RvcjxmbG9hdCwgc3RkOjpfXzI6OmFsbG9jYXRvcjxmbG9hdD4+IGNvbnN0JiwgdW5zaWduZWQgaW50Pjo6aW52b2tlKHZvaWQgKGVtc2NyaXB0ZW46OkJydXRlZm9yY2VTZWFyY2g6OiogY29uc3QmKShzdGQ6Ol9fMjo6dmVjdG9yPGZsb2F0LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPGZsb2F0Pj4gY29uc3QmLCB1bnNpZ25lZCBpbnQpLCBlbXNjcmlwdGVuOjpCcnV0ZWZvcmNlU2VhcmNoKiwgZW1zY3JpcHRlbjo6X0VNX1ZBTCosIHVuc2lnbmVkIGludClXN2Vtc2NyaXB0ZW46OkJydXRlZm9yY2VTZWFyY2g6OnJlbW92ZVBvaW50KHVuc2lnbmVkIGludClYggFlbXNjcmlwdGVuOjpCcnV0ZWZvcmNlU2VhcmNoOjpzZWFyY2hLbm4oc3RkOjpfXzI6OnZlY3RvcjxmbG9hdCwgc3RkOjpfXzI6OmFsbG9jYXRvcjxmbG9hdD4+IGNvbnN0JiwgdW5zaWduZWQgaW50LCBlbXNjcmlwdGVuOjp2YWwpWbIEZW1zY3JpcHRlbjo6aW50ZXJuYWw6Ok1ldGhvZEludm9rZXI8ZW1zY3JpcHRlbjo6dmFsIChlbXNjcmlwdGVuOjpCcnV0ZWZvcmNlU2VhcmNoOjoqKShzdGQ6Ol9fMjo6dmVjdG9yPGZsb2F0LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPGZsb2F0Pj4gY29uc3QmLCB1bnNpZ25lZCBpbnQsIGVtc2NyaXB0ZW46OnZhbCksIGVtc2NyaXB0ZW46OnZhbCwgZW1zY3JpcHRlbjo6QnJ1dGVmb3JjZVNlYXJjaCosIHN0ZDo6X18yOjp2ZWN0b3I8ZmxvYXQsIHN0ZDo6X18yOjphbGxvY2F0b3I8ZmxvYXQ+PiBjb25zdCYsIHVuc2lnbmVkIGludCwgZW1zY3JpcHRlbjo6dmFsPjo6aW52b2tlKGVtc2NyaXB0ZW46OnZhbCAoZW1zY3JpcHRlbjo6QnJ1dGVmb3JjZVNlYXJjaDo6KiBjb25zdCYpKHN0ZDo6X18yOjp2ZWN0b3I8ZmxvYXQsIHN0ZDo6X18yOjphbGxvY2F0b3I8ZmxvYXQ+PiBjb25zdCYsIHVuc2lnbmVkIGludCwgZW1zY3JpcHRlbjo6dmFsKSwgZW1zY3JpcHRlbjo6QnJ1dGVmb3JjZVNlYXJjaCosIGVtc2NyaXB0ZW46Ol9FTV9WQUwqLCB1bnNpZ25lZCBpbnQsIGVtc2NyaXB0ZW46Ol9FTV9WQUwqKVouZW1zY3JpcHRlbjo6QnJ1dGVmb3JjZVNlYXJjaDo6Z2V0TWF4RWxlbWVudHMoKVsvZW1zY3JpcHRlbjo6QnJ1dGVmb3JjZVNlYXJjaDo6Z2V0Q3VycmVudENvdW50KClcanZvaWQgY29uc3QqIGVtc2NyaXB0ZW46OmludGVybmFsOjpnZXRBY3R1YWxUeXBlPGVtc2NyaXB0ZW46OkhpZXJhcmNoaWNhbE5TVz4oZW1zY3JpcHRlbjo6SGllcmFyY2hpY2FsTlNXKildZHZvaWQgZW1zY3JpcHRlbjo6aW50ZXJuYWw6OnJhd19kZXN0cnVjdG9yPGVtc2NyaXB0ZW46OkhpZXJhcmNoaWNhbE5TVz4oZW1zY3JpcHRlbjo6SGllcmFyY2hpY2FsTlNXKilenQZlbXNjcmlwdGVuOjppbnRlcm5hbDo6SW52b2tlcjxlbXNjcmlwdGVuOjpIaWVyYXJjaGljYWxOU1cqLCBzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPiwgc3RkOjpfXzI6OmFsbG9jYXRvcjxjaGFyPj4gY29uc3QmLCB1bnNpZ25lZCBpbnQmJiwgc3RkOjpfXzI6OmJhc2ljX3N0cmluZzxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4sIHN0ZDo6X18yOjphbGxvY2F0b3I8Y2hhcj4+IGNvbnN0Jj46Omludm9rZShlbXNjcmlwdGVuOjpIaWVyYXJjaGljYWxOU1cqICgqKShzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPiwgc3RkOjpfXzI6OmFsbG9jYXRvcjxjaGFyPj4gY29uc3QmLCB1bnNpZ25lZCBpbnQmJiwgc3RkOjpfXzI6OmJhc2ljX3N0cmluZzxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4sIHN0ZDo6X18yOjphbGxvY2F0b3I8Y2hhcj4+IGNvbnN0JiksIGVtc2NyaXB0ZW46OmludGVybmFsOjpCaW5kaW5nVHlwZTxzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPiwgc3RkOjpfXzI6OmFsbG9jYXRvcjxjaGFyPj4sIHZvaWQ+OjondW5uYW1lZCcqLCB1bnNpZ25lZCBpbnQsIGVtc2NyaXB0ZW46OmludGVybmFsOjpCaW5kaW5nVHlwZTxzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPiwgc3RkOjpfXzI6OmFsbG9jYXRvcjxjaGFyPj4sIHZvaWQ+OjondW5uYW1lZCcqKV/uA2Vtc2NyaXB0ZW46OkhpZXJhcmNoaWNhbE5TVyogZW1zY3JpcHRlbjo6aW50ZXJuYWw6Om9wZXJhdG9yX25ldzxlbXNjcmlwdGVuOjpIaWVyYXJjaGljYWxOU1csIHN0ZDo6X18yOjpiYXNpY19zdHJpbmc8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPGNoYXI+PiBjb25zdCYsIHVuc2lnbmVkIGludCwgc3RkOjpfXzI6OmJhc2ljX3N0cmluZzxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4sIHN0ZDo6X18yOjphbGxvY2F0b3I8Y2hhcj4+IGNvbnN0Jj4oc3RkOjpfXzI6OmJhc2ljX3N0cmluZzxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4sIHN0ZDo6X18yOjphbGxvY2F0b3I8Y2hhcj4+IGNvbnN0JiwgdW5zaWduZWQgaW50JiYsIHN0ZDo6X18yOjpiYXNpY19zdHJpbmc8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPGNoYXI+PiBjb25zdCYpYF5lbXNjcmlwdGVuOjpIaWVyYXJjaGljYWxOU1c6OmluaXRJbmRleCh1bnNpZ25lZCBpbnQsIHVuc2lnbmVkIGludCwgdW5zaWduZWQgaW50LCB1bnNpZ25lZCBpbnQpYaIDZW1zY3JpcHRlbjo6aW50ZXJuYWw6Ok1ldGhvZEludm9rZXI8dm9pZCAoZW1zY3JpcHRlbjo6SGllcmFyY2hpY2FsTlNXOjoqKSh1bnNpZ25lZCBpbnQsIHVuc2lnbmVkIGludCwgdW5zaWduZWQgaW50LCB1bnNpZ25lZCBpbnQpLCB2b2lkLCBlbXNjcmlwdGVuOjpIaWVyYXJjaGljYWxOU1cqLCB1bnNpZ25lZCBpbnQsIHVuc2lnbmVkIGludCwgdW5zaWduZWQgaW50LCB1bnNpZ25lZCBpbnQ+OjppbnZva2Uodm9pZCAoZW1zY3JpcHRlbjo6SGllcmFyY2hpY2FsTlNXOjoqIGNvbnN0JikodW5zaWduZWQgaW50LCB1bnNpZ25lZCBpbnQsIHVuc2lnbmVkIGludCwgdW5zaWduZWQgaW50KSwgZW1zY3JpcHRlbjo6SGllcmFyY2hpY2FsTlNXKiwgdW5zaWduZWQgaW50LCB1bnNpZ25lZCBpbnQsIHVuc2lnbmVkIGludCwgdW5zaWduZWQgaW50KWKRAWVtc2NyaXB0ZW46OkhpZXJhcmNoaWNhbE5TVzo6cmVhZEluZGV4KHN0ZDo6X18yOjpiYXNpY19zdHJpbmc8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPGNoYXI+PiBjb25zdCYsIHVuc2lnbmVkIGludCljnAVlbXNjcmlwdGVuOjppbnRlcm5hbDo6TWV0aG9kSW52b2tlcjx2b2lkIChlbXNjcmlwdGVuOjpIaWVyYXJjaGljYWxOU1c6OiopKHN0ZDo6X18yOjpiYXNpY19zdHJpbmc8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPGNoYXI+PiBjb25zdCYsIHVuc2lnbmVkIGludCksIHZvaWQsIGVtc2NyaXB0ZW46OkhpZXJhcmNoaWNhbE5TVyosIHN0ZDo6X18yOjpiYXNpY19zdHJpbmc8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPGNoYXI+PiBjb25zdCYsIHVuc2lnbmVkIGludD46Omludm9rZSh2b2lkIChlbXNjcmlwdGVuOjpIaWVyYXJjaGljYWxOU1c6OiogY29uc3QmKShzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPiwgc3RkOjpfXzI6OmFsbG9jYXRvcjxjaGFyPj4gY29uc3QmLCB1bnNpZ25lZCBpbnQpLCBlbXNjcmlwdGVuOjpIaWVyYXJjaGljYWxOU1cqLCBlbXNjcmlwdGVuOjppbnRlcm5hbDo6QmluZGluZ1R5cGU8c3RkOjpfXzI6OmJhc2ljX3N0cmluZzxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4sIHN0ZDo6X18yOjphbGxvY2F0b3I8Y2hhcj4+LCB2b2lkPjo6J3VubmFtZWQnKiwgdW5zaWduZWQgaW50KWSEAWVtc2NyaXB0ZW46OkhpZXJhcmNoaWNhbE5TVzo6d3JpdGVJbmRleChzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPiwgc3RkOjpfXzI6OmFsbG9jYXRvcjxjaGFyPj4gY29uc3QmKWU2ZW1zY3JpcHRlbjo6SGllcmFyY2hpY2FsTlNXOjpyZXNpemVJbmRleCh1bnNpZ25lZCBpbnQpZjNlbXNjcmlwdGVuOjpIaWVyYXJjaGljYWxOU1c6OmdldFBvaW50KHVuc2lnbmVkIGludClnmwJlbXNjcmlwdGVuOjppbnRlcm5hbDo6TWV0aG9kSW52b2tlcjxlbXNjcmlwdGVuOjp2YWwgKGVtc2NyaXB0ZW46OkhpZXJhcmNoaWNhbE5TVzo6KikodW5zaWduZWQgaW50KSwgZW1zY3JpcHRlbjo6dmFsLCBlbXNjcmlwdGVuOjpIaWVyYXJjaGljYWxOU1cqLCB1bnNpZ25lZCBpbnQ+OjppbnZva2UoZW1zY3JpcHRlbjo6dmFsIChlbXNjcmlwdGVuOjpIaWVyYXJjaGljYWxOU1c6OiogY29uc3QmKSh1bnNpZ25lZCBpbnQpLCBlbXNjcmlwdGVuOjpIaWVyYXJjaGljYWxOU1cqLCB1bnNpZ25lZCBpbnQpaHVlbXNjcmlwdGVuOjpIaWVyYXJjaGljYWxOU1c6OmFkZFBvaW50KHN0ZDo6X18yOjp2ZWN0b3I8ZmxvYXQsIHN0ZDo6X18yOjphbGxvY2F0b3I8ZmxvYXQ+PiBjb25zdCYsIHVuc2lnbmVkIGludCwgYm9vbClp3ANlbXNjcmlwdGVuOjppbnRlcm5hbDo6TWV0aG9kSW52b2tlcjx2b2lkIChlbXNjcmlwdGVuOjpIaWVyYXJjaGljYWxOU1c6OiopKHN0ZDo6X18yOjp2ZWN0b3I8ZmxvYXQsIHN0ZDo6X18yOjphbGxvY2F0b3I8ZmxvYXQ+PiBjb25zdCYsIHVuc2lnbmVkIGludCwgYm9vbCksIHZvaWQsIGVtc2NyaXB0ZW46OkhpZXJhcmNoaWNhbE5TVyosIHN0ZDo6X18yOjp2ZWN0b3I8ZmxvYXQsIHN0ZDo6X18yOjphbGxvY2F0b3I8ZmxvYXQ+PiBjb25zdCYsIHVuc2lnbmVkIGludCwgYm9vbD46Omludm9rZSh2b2lkIChlbXNjcmlwdGVuOjpIaWVyYXJjaGljYWxOU1c6OiogY29uc3QmKShzdGQ6Ol9fMjo6dmVjdG9yPGZsb2F0LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPGZsb2F0Pj4gY29uc3QmLCB1bnNpZ25lZCBpbnQsIGJvb2wpLCBlbXNjcmlwdGVuOjpIaWVyYXJjaGljYWxOU1cqLCBlbXNjcmlwdGVuOjpfRU1fVkFMKiwgdW5zaWduZWQgaW50LCBib29sKWqOAmVtc2NyaXB0ZW46OkhpZXJhcmNoaWNhbE5TVzo6YWRkUG9pbnRzKHN0ZDo6X18yOjp2ZWN0b3I8c3RkOjpfXzI6OnZlY3RvcjxmbG9hdCwgc3RkOjpfXzI6OmFsbG9jYXRvcjxmbG9hdD4+LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPHN0ZDo6X18yOjp2ZWN0b3I8ZmxvYXQsIHN0ZDo6X18yOjphbGxvY2F0b3I8ZmxvYXQ+Pj4+IGNvbnN0Jiwgc3RkOjpfXzI6OnZlY3Rvcjx1bnNpZ25lZCBpbnQsIHN0ZDo6X18yOjphbGxvY2F0b3I8dW5zaWduZWQgaW50Pj4gY29uc3QmLCBib29sKWusB2Vtc2NyaXB0ZW46OmludGVybmFsOjpNZXRob2RJbnZva2VyPHZvaWQgKGVtc2NyaXB0ZW46OkhpZXJhcmNoaWNhbE5TVzo6Kikoc3RkOjpfXzI6OnZlY3RvcjxzdGQ6Ol9fMjo6dmVjdG9yPGZsb2F0LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPGZsb2F0Pj4sIHN0ZDo6X18yOjphbGxvY2F0b3I8c3RkOjpfXzI6OnZlY3RvcjxmbG9hdCwgc3RkOjpfXzI6OmFsbG9jYXRvcjxmbG9hdD4+Pj4gY29uc3QmLCBzdGQ6Ol9fMjo6dmVjdG9yPHVuc2lnbmVkIGludCwgc3RkOjpfXzI6OmFsbG9jYXRvcjx1bnNpZ25lZCBpbnQ+PiBjb25zdCYsIGJvb2wpLCB2b2lkLCBlbXNjcmlwdGVuOjpIaWVyYXJjaGljYWxOU1cqLCBzdGQ6Ol9fMjo6dmVjdG9yPHN0ZDo6X18yOjp2ZWN0b3I8ZmxvYXQsIHN0ZDo6X18yOjphbGxvY2F0b3I8ZmxvYXQ+Piwgc3RkOjpfXzI6OmFsbG9jYXRvcjxzdGQ6Ol9fMjo6dmVjdG9yPGZsb2F0LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPGZsb2F0Pj4+PiBjb25zdCYsIHN0ZDo6X18yOjp2ZWN0b3I8dW5zaWduZWQgaW50LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPHVuc2lnbmVkIGludD4+IGNvbnN0JiwgYm9vbD46Omludm9rZSh2b2lkIChlbXNjcmlwdGVuOjpIaWVyYXJjaGljYWxOU1c6OiogY29uc3QmKShzdGQ6Ol9fMjo6dmVjdG9yPHN0ZDo6X18yOjp2ZWN0b3I8ZmxvYXQsIHN0ZDo6X18yOjphbGxvY2F0b3I8ZmxvYXQ+Piwgc3RkOjpfXzI6OmFsbG9jYXRvcjxzdGQ6Ol9fMjo6dmVjdG9yPGZsb2F0LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPGZsb2F0Pj4+PiBjb25zdCYsIHN0ZDo6X18yOjp2ZWN0b3I8dW5zaWduZWQgaW50LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPHVuc2lnbmVkIGludD4+IGNvbnN0JiwgYm9vbCksIGVtc2NyaXB0ZW46OkhpZXJhcmNoaWNhbE5TVyosIGVtc2NyaXB0ZW46Ol9FTV9WQUwqLCBlbXNjcmlwdGVuOjpfRU1fVkFMKiwgYm9vbClswwFlbXNjcmlwdGVuOjpIaWVyYXJjaGljYWxOU1c6OmFkZEl0ZW1zKHN0ZDo6X18yOjp2ZWN0b3I8c3RkOjpfXzI6OnZlY3RvcjxmbG9hdCwgc3RkOjpfXzI6OmFsbG9jYXRvcjxmbG9hdD4+LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPHN0ZDo6X18yOjp2ZWN0b3I8ZmxvYXQsIHN0ZDo6X18yOjphbGxvY2F0b3I8ZmxvYXQ+Pj4+IGNvbnN0JiwgYm9vbClt7wZlbXNjcmlwdGVuOjppbnRlcm5hbDo6TWV0aG9kSW52b2tlcjxzdGQ6Ol9fMjo6dmVjdG9yPHVuc2lnbmVkIGludCwgc3RkOjpfXzI6OmFsbG9jYXRvcjx1bnNpZ25lZCBpbnQ+PiAoZW1zY3JpcHRlbjo6SGllcmFyY2hpY2FsTlNXOjoqKShzdGQ6Ol9fMjo6dmVjdG9yPHN0ZDo6X18yOjp2ZWN0b3I8ZmxvYXQsIHN0ZDo6X18yOjphbGxvY2F0b3I8ZmxvYXQ+Piwgc3RkOjpfXzI6OmFsbG9jYXRvcjxzdGQ6Ol9fMjo6dmVjdG9yPGZsb2F0LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPGZsb2F0Pj4+PiBjb25zdCYsIGJvb2wpLCBzdGQ6Ol9fMjo6dmVjdG9yPHVuc2lnbmVkIGludCwgc3RkOjpfXzI6OmFsbG9jYXRvcjx1bnNpZ25lZCBpbnQ+PiwgZW1zY3JpcHRlbjo6SGllcmFyY2hpY2FsTlNXKiwgc3RkOjpfXzI6OnZlY3RvcjxzdGQ6Ol9fMjo6dmVjdG9yPGZsb2F0LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPGZsb2F0Pj4sIHN0ZDo6X18yOjphbGxvY2F0b3I8c3RkOjpfXzI6OnZlY3RvcjxmbG9hdCwgc3RkOjpfXzI6OmFsbG9jYXRvcjxmbG9hdD4+Pj4gY29uc3QmLCBib29sPjo6aW52b2tlKHN0ZDo6X18yOjp2ZWN0b3I8dW5zaWduZWQgaW50LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPHVuc2lnbmVkIGludD4+IChlbXNjcmlwdGVuOjpIaWVyYXJjaGljYWxOU1c6OiogY29uc3QmKShzdGQ6Ol9fMjo6dmVjdG9yPHN0ZDo6X18yOjp2ZWN0b3I8ZmxvYXQsIHN0ZDo6X18yOjphbGxvY2F0b3I8ZmxvYXQ+Piwgc3RkOjpfXzI6OmFsbG9jYXRvcjxzdGQ6Ol9fMjo6dmVjdG9yPGZsb2F0LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPGZsb2F0Pj4+PiBjb25zdCYsIGJvb2wpLCBlbXNjcmlwdGVuOjpIaWVyYXJjaGljYWxOU1cqLCBlbXNjcmlwdGVuOjpfRU1fVkFMKiwgYm9vbCluLGVtc2NyaXB0ZW46OkhpZXJhcmNoaWNhbE5TVzo6Z2V0VXNlZExhYmVscygpb/0CZW1zY3JpcHRlbjo6aW50ZXJuYWw6Ok1ldGhvZEludm9rZXI8c3RkOjpfXzI6OnZlY3Rvcjx1bnNpZ25lZCBpbnQsIHN0ZDo6X18yOjphbGxvY2F0b3I8dW5zaWduZWQgaW50Pj4gKGVtc2NyaXB0ZW46OkhpZXJhcmNoaWNhbE5TVzo6KikoKSwgc3RkOjpfXzI6OnZlY3Rvcjx1bnNpZ25lZCBpbnQsIHN0ZDo6X18yOjphbGxvY2F0b3I8dW5zaWduZWQgaW50Pj4sIGVtc2NyaXB0ZW46OkhpZXJhcmNoaWNhbE5TVyo+OjppbnZva2Uoc3RkOjpfXzI6OnZlY3Rvcjx1bnNpZ25lZCBpbnQsIHN0ZDo6X18yOjphbGxvY2F0b3I8dW5zaWduZWQgaW50Pj4gKGVtc2NyaXB0ZW46OkhpZXJhcmNoaWNhbE5TVzo6KiBjb25zdCYpKCksIGVtc2NyaXB0ZW46OkhpZXJhcmNoaWNhbE5TVyopcC9lbXNjcmlwdGVuOjpIaWVyYXJjaGljYWxOU1c6OmdldERlbGV0ZWRMYWJlbHMoKXEtZW1zY3JpcHRlbjo6SGllcmFyY2hpY2FsTlNXOjpnZXRNYXhFbGVtZW50cygpcjVlbXNjcmlwdGVuOjpIaWVyYXJjaGljYWxOU1c6Om1hcmtEZWxldGUodW5zaWduZWQgaW50KXN2ZW1zY3JpcHRlbjo6SGllcmFyY2hpY2FsTlNXOjptYXJrRGVsZXRlSXRlbXMoc3RkOjpfXzI6OnZlY3Rvcjx1bnNpZ25lZCBpbnQsIHN0ZDo6X18yOjphbGxvY2F0b3I8dW5zaWduZWQgaW50Pj4gY29uc3QmKXS2A2Vtc2NyaXB0ZW46OmludGVybmFsOjpNZXRob2RJbnZva2VyPHZvaWQgKGVtc2NyaXB0ZW46OkhpZXJhcmNoaWNhbE5TVzo6Kikoc3RkOjpfXzI6OnZlY3Rvcjx1bnNpZ25lZCBpbnQsIHN0ZDo6X18yOjphbGxvY2F0b3I8dW5zaWduZWQgaW50Pj4gY29uc3QmKSwgdm9pZCwgZW1zY3JpcHRlbjo6SGllcmFyY2hpY2FsTlNXKiwgc3RkOjpfXzI6OnZlY3Rvcjx1bnNpZ25lZCBpbnQsIHN0ZDo6X18yOjphbGxvY2F0b3I8dW5zaWduZWQgaW50Pj4gY29uc3QmPjo6aW52b2tlKHZvaWQgKGVtc2NyaXB0ZW46OkhpZXJhcmNoaWNhbE5TVzo6KiBjb25zdCYpKHN0ZDo6X18yOjp2ZWN0b3I8dW5zaWduZWQgaW50LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPHVuc2lnbmVkIGludD4+IGNvbnN0JiksIGVtc2NyaXB0ZW46OkhpZXJhcmNoaWNhbE5TVyosIGVtc2NyaXB0ZW46Ol9FTV9WQUwqKXU3ZW1zY3JpcHRlbjo6SGllcmFyY2hpY2FsTlNXOjp1bm1hcmtEZWxldGUodW5zaWduZWQgaW50KXY0ZW1zY3JpcHRlbjo6SGllcmFyY2hpY2FsTlNXOjpnZXRDdXJyZW50Q291bnQoKSBjb25zdHcwZW1zY3JpcHRlbjo6SGllcmFyY2hpY2FsTlNXOjpnZXRFZlNlYXJjaCgpIGNvbnN0eDZlbXNjcmlwdGVuOjpIaWVyYXJjaGljYWxOU1c6OnNldEVmU2VhcmNoKHVuc2lnbmVkIGludCl5gQFlbXNjcmlwdGVuOjpIaWVyYXJjaGljYWxOU1c6OnNlYXJjaEtubihzdGQ6Ol9fMjo6dmVjdG9yPGZsb2F0LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPGZsb2F0Pj4gY29uc3QmLCB1bnNpZ25lZCBpbnQsIGVtc2NyaXB0ZW46OnZhbCl6R2Vtc2NyaXB0ZW46OmludGVybmFsOjpJbnZva2VyPHZvaWQsIGJvb2w+OjppbnZva2Uodm9pZCAoKikoYm9vbCksIGJvb2wpe4IBdm9pZCBjb25zdCogZW1zY3JpcHRlbjo6aW50ZXJuYWw6OmdldEFjdHVhbFR5cGU8ZW1zY3JpcHRlbjo6RW1zY3JpcHRlbkZpbGVTeXN0ZW1NYW5hZ2VyPihlbXNjcmlwdGVuOjpFbXNjcmlwdGVuRmlsZVN5c3RlbU1hbmFnZXIqKXx8dm9pZCBlbXNjcmlwdGVuOjppbnRlcm5hbDo6cmF3X2Rlc3RydWN0b3I8ZW1zY3JpcHRlbjo6RW1zY3JpcHRlbkZpbGVTeXN0ZW1NYW5hZ2VyPihlbXNjcmlwdGVuOjpFbXNjcmlwdGVuRmlsZVN5c3RlbU1hbmFnZXIqKX1/ZW1zY3JpcHRlbjo6aW50ZXJuYWw6Okludm9rZXI8ZW1zY3JpcHRlbjo6RW1zY3JpcHRlbkZpbGVTeXN0ZW1NYW5hZ2VyKj46Omludm9rZShlbXNjcmlwdGVuOjpFbXNjcmlwdGVuRmlsZVN5c3RlbU1hbmFnZXIqICgqKSgpKX52ZW1zY3JpcHRlbjo6RW1zY3JpcHRlbkZpbGVTeXN0ZW1NYW5hZ2VyKiBlbXNjcmlwdGVuOjppbnRlcm5hbDo6b3BlcmF0b3JfbmV3PGVtc2NyaXB0ZW46OkVtc2NyaXB0ZW5GaWxlU3lzdGVtTWFuYWdlcj4oKX/6AmVtc2NyaXB0ZW46OmludGVybmFsOjpJbnZva2VyPHZvaWQsIHN0ZDo6X18yOjpiYXNpY19zdHJpbmc8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPGNoYXI+PiBjb25zdCY+OjppbnZva2Uodm9pZCAoKikoc3RkOjpfXzI6OmJhc2ljX3N0cmluZzxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4sIHN0ZDo6X18yOjphbGxvY2F0b3I8Y2hhcj4+IGNvbnN0JiksIGVtc2NyaXB0ZW46OmludGVybmFsOjpCaW5kaW5nVHlwZTxzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPiwgc3RkOjpfXzI6OmFsbG9jYXRvcjxjaGFyPj4sIHZvaWQ+OjondW5uYW1lZCcqKYABmgFlbXNjcmlwdGVuOjpFbXNjcmlwdGVuRmlsZVN5c3RlbU1hbmFnZXI6OmluaXRpYWxpemVGaWxlU3lzdGVtKHN0ZDo6X18yOjpiYXNpY19zdHJpbmc8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPGNoYXI+PiBjb25zdCYpgQE4ZW1zY3JpcHRlbjo6RW1zY3JpcHRlbkZpbGVTeXN0ZW1NYW5hZ2VyOjppc0luaXRpYWxpemVkKCmCAX9lbXNjcmlwdGVuOjppbnRlcm5hbDo6SW52b2tlcjx2b2lkLCBib29sLCBlbXNjcmlwdGVuOjp2YWw+OjppbnZva2Uodm9pZCAoKikoYm9vbCwgZW1zY3JpcHRlbjo6dmFsKSwgYm9vbCwgZW1zY3JpcHRlbjo6X0VNX1ZBTCopgwFGZW1zY3JpcHRlbjo6RW1zY3JpcHRlbkZpbGVTeXN0ZW1NYW5hZ2VyOjpzeW5jRlMoYm9vbCwgZW1zY3JpcHRlbjo6dmFsKYQBO2Vtc2NyaXB0ZW46OkVtc2NyaXB0ZW5GaWxlU3lzdGVtTWFuYWdlcjo6c2V0RGVidWdMb2dzKGJvb2wphQEzZW1zY3JpcHRlbjo6RW1zY3JpcHRlbkZpbGVTeXN0ZW1NYW5hZ2VyOjppc1N5bmNlZCgphgH6AmVtc2NyaXB0ZW46OmludGVybmFsOjpJbnZva2VyPGJvb2wsIHN0ZDo6X18yOjpiYXNpY19zdHJpbmc8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPGNoYXI+PiBjb25zdCY+OjppbnZva2UoYm9vbCAoKikoc3RkOjpfXzI6OmJhc2ljX3N0cmluZzxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4sIHN0ZDo6X18yOjphbGxvY2F0b3I8Y2hhcj4+IGNvbnN0JiksIGVtc2NyaXB0ZW46OmludGVybmFsOjpCaW5kaW5nVHlwZTxzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPiwgc3RkOjpfXzI6OmFsbG9jYXRvcjxjaGFyPj4sIHZvaWQ+OjondW5uYW1lZCcqKYcBlQFlbXNjcmlwdGVuOjpFbXNjcmlwdGVuRmlsZVN5c3RlbU1hbmFnZXI6OmNoZWNrRmlsZUV4aXN0cyhzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPiwgc3RkOjpfXzI6OmFsbG9jYXRvcjxjaGFyPj4gY29uc3QmKYgBYmhuc3dsaWI6OkJydXRlZm9yY2VTZWFyY2g8ZmxvYXQ+OjpCcnV0ZWZvcmNlU2VhcmNoKGhuc3dsaWI6OlNwYWNlSW50ZXJmYWNlPGZsb2F0PiosIHVuc2lnbmVkIGxvbmcpiQGpAWhuc3dsaWI6OkJydXRlZm9yY2VTZWFyY2g8ZmxvYXQ+Ojpsb2FkSW5kZXgoc3RkOjpfXzI6OmJhc2ljX3N0cmluZzxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4sIHN0ZDo6X18yOjphbGxvY2F0b3I8Y2hhcj4+IGNvbnN0JiwgaG5zd2xpYjo6U3BhY2VJbnRlcmZhY2U8ZmxvYXQ+KimKAd4Bc3RkOjpfXzI6OnVub3JkZXJlZF9tYXA8dW5zaWduZWQgbG9uZywgdW5zaWduZWQgbG9uZywgc3RkOjpfXzI6Omhhc2g8dW5zaWduZWQgbG9uZz4sIHN0ZDo6X18yOjplcXVhbF90bzx1bnNpZ25lZCBsb25nPiwgc3RkOjpfXzI6OmFsbG9jYXRvcjxzdGQ6Ol9fMjo6cGFpcjx1bnNpZ25lZCBsb25nIGNvbnN0LCB1bnNpZ25lZCBsb25nPj4+Ojp+dW5vcmRlcmVkX21hcFthYmk6djE1MDA3XSgpiwHYAXN0ZDo6X18yOjpiYXNpY19zdHJpbmc8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPGNoYXI+Pjo6ZmluZFthYmk6djE1MDA3XShzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPiwgc3RkOjpfXzI6OmFsbG9jYXRvcjxjaGFyPj4gY29uc3QmLCB1bnNpZ25lZCBsb25nKSBjb25zdIwBmgdzdGQ6Ol9fMjo6cGFpcjxzdGQ6Ol9fMjo6X19oYXNoX2l0ZXJhdG9yPHN0ZDo6X18yOjpfX2hhc2hfbm9kZTxzdGQ6Ol9fMjo6X19oYXNoX3ZhbHVlX3R5cGU8dW5zaWduZWQgbG9uZywgdW5zaWduZWQgbG9uZz4sIHZvaWQqPio+LCBib29sPiBzdGQ6Ol9fMjo6X19oYXNoX3RhYmxlPHN0ZDo6X18yOjpfX2hhc2hfdmFsdWVfdHlwZTx1bnNpZ25lZCBsb25nLCB1bnNpZ25lZCBsb25nPiwgc3RkOjpfXzI6Ol9fdW5vcmRlcmVkX21hcF9oYXNoZXI8dW5zaWduZWQgbG9uZywgc3RkOjpfXzI6Ol9faGFzaF92YWx1ZV90eXBlPHVuc2lnbmVkIGxvbmcsIHVuc2lnbmVkIGxvbmc+LCBzdGQ6Ol9fMjo6aGFzaDx1bnNpZ25lZCBsb25nPiwgc3RkOjpfXzI6OmVxdWFsX3RvPHVuc2lnbmVkIGxvbmc+LCB0cnVlPiwgc3RkOjpfXzI6Ol9fdW5vcmRlcmVkX21hcF9lcXVhbDx1bnNpZ25lZCBsb25nLCBzdGQ6Ol9fMjo6X19oYXNoX3ZhbHVlX3R5cGU8dW5zaWduZWQgbG9uZywgdW5zaWduZWQgbG9uZz4sIHN0ZDo6X18yOjplcXVhbF90bzx1bnNpZ25lZCBsb25nPiwgc3RkOjpfXzI6Omhhc2g8dW5zaWduZWQgbG9uZz4sIHRydWU+LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPHN0ZDo6X18yOjpfX2hhc2hfdmFsdWVfdHlwZTx1bnNpZ25lZCBsb25nLCB1bnNpZ25lZCBsb25nPj4+OjpfX2VtcGxhY2VfdW5pcXVlX2tleV9hcmdzPHVuc2lnbmVkIGxvbmcsIHN0ZDo6X18yOjpwaWVjZXdpc2VfY29uc3RydWN0X3QgY29uc3QmLCBzdGQ6Ol9fMjo6dHVwbGU8dW5zaWduZWQgbG9uZyBjb25zdCY+LCBzdGQ6Ol9fMjo6dHVwbGU8Pj4odW5zaWduZWQgbG9uZyBjb25zdCYsIHN0ZDo6X18yOjpwaWVjZXdpc2VfY29uc3RydWN0X3QgY29uc3QmLCBzdGQ6Ol9fMjo6dHVwbGU8dW5zaWduZWQgbG9uZyBjb25zdCY+JiYsIHN0ZDo6X18yOjp0dXBsZTw+JiYpjQHLBHVuc2lnbmVkIGxvbmcgc3RkOjpfXzI6Ol9faGFzaF90YWJsZTxzdGQ6Ol9fMjo6X19oYXNoX3ZhbHVlX3R5cGU8dW5zaWduZWQgbG9uZywgdW5zaWduZWQgbG9uZz4sIHN0ZDo6X18yOjpfX3Vub3JkZXJlZF9tYXBfaGFzaGVyPHVuc2lnbmVkIGxvbmcsIHN0ZDo6X18yOjpfX2hhc2hfdmFsdWVfdHlwZTx1bnNpZ25lZCBsb25nLCB1bnNpZ25lZCBsb25nPiwgc3RkOjpfXzI6Omhhc2g8dW5zaWduZWQgbG9uZz4sIHN0ZDo6X18yOjplcXVhbF90bzx1bnNpZ25lZCBsb25nPiwgdHJ1ZT4sIHN0ZDo6X18yOjpfX3Vub3JkZXJlZF9tYXBfZXF1YWw8dW5zaWduZWQgbG9uZywgc3RkOjpfXzI6Ol9faGFzaF92YWx1ZV90eXBlPHVuc2lnbmVkIGxvbmcsIHVuc2lnbmVkIGxvbmc+LCBzdGQ6Ol9fMjo6ZXF1YWxfdG88dW5zaWduZWQgbG9uZz4sIHN0ZDo6X18yOjpoYXNoPHVuc2lnbmVkIGxvbmc+LCB0cnVlPiwgc3RkOjpfXzI6OmFsbG9jYXRvcjxzdGQ6Ol9fMjo6X19oYXNoX3ZhbHVlX3R5cGU8dW5zaWduZWQgbG9uZywgdW5zaWduZWQgbG9uZz4+Pjo6X19lcmFzZV91bmlxdWU8dW5zaWduZWQgbG9uZz4odW5zaWduZWQgbG9uZyBjb25zdCYpjgFAc3RkOjppbnZhbGlkX2FyZ3VtZW50OjppbnZhbGlkX2FyZ3VtZW50W2FiaTp2MTUwMDddKGNoYXIgY29uc3QqKY8BP3ZvaWQgZW1zY3JpcHRlbjo6dmFsOjpzZXQ8aW50LCBmbG9hdD4oaW50IGNvbnN0JiwgZmxvYXQgY29uc3QmKZABT3ZvaWQgZW1zY3JpcHRlbjo6dmFsOjpzZXQ8aW50LCB1bnNpZ25lZCBsb25nPihpbnQgY29uc3QmLCB1bnNpZ25lZCBsb25nIGNvbnN0JimRAdMDdm9pZCBzdGQ6Ol9fMjo6X19wb3BfaGVhcFthYmk6djE1MDA3XTxzdGQ6Ol9fMjo6X0NsYXNzaWNBbGdQb2xpY3ksIHN0ZDo6X18yOjpsZXNzPHN0ZDo6X18yOjpwYWlyPGZsb2F0LCB1bnNpZ25lZCBsb25nPj4sIHN0ZDo6X18yOjpfX3dyYXBfaXRlcjxzdGQ6Ol9fMjo6cGFpcjxmbG9hdCwgdW5zaWduZWQgbG9uZz4qPj4oc3RkOjpfXzI6Ol9fd3JhcF9pdGVyPHN0ZDo6X18yOjpwYWlyPGZsb2F0LCB1bnNpZ25lZCBsb25nPio+LCBzdGQ6Ol9fMjo6X193cmFwX2l0ZXI8c3RkOjpfXzI6OnBhaXI8ZmxvYXQsIHVuc2lnbmVkIGxvbmc+Kj4sIHN0ZDo6X18yOjpsZXNzPHN0ZDo6X18yOjpwYWlyPGZsb2F0LCB1bnNpZ25lZCBsb25nPj4mLCBzdGQ6Ol9fMjo6aXRlcmF0b3JfdHJhaXRzPHN0ZDo6X18yOjpfX3dyYXBfaXRlcjxzdGQ6Ol9fMjo6cGFpcjxmbG9hdCwgdW5zaWduZWQgbG9uZz4qPj46OmRpZmZlcmVuY2VfdHlwZSmSAZMBaG5zd2xpYjo6SGllcmFyY2hpY2FsTlNXPGZsb2F0Pjo6SGllcmFyY2hpY2FsTlNXKGhuc3dsaWI6OlNwYWNlSW50ZXJmYWNlPGZsb2F0PiosIHVuc2lnbmVkIGxvbmcsIHVuc2lnbmVkIGxvbmcsIHVuc2lnbmVkIGxvbmcsIHVuc2lnbmVkIGxvbmcsIGJvb2wpkwEwZW1zY3JpcHRlbjo6SGllcmFyY2hpY2FsTlNXOjp1cGRhdGVMYWJlbENhY2hlcygplAE7aG5zd2xpYjo6SGllcmFyY2hpY2FsTlNXPGZsb2F0Pjo6cmVzaXplSW5kZXgodW5zaWduZWQgbG9uZymVASxlbXNjcmlwdGVuOjpIaWVyYXJjaGljYWxOU1c6OmF1dG9TYXZlSW5kZXgoKZYBf3N0ZDo6X18yOjp2ZWN0b3I8ZmxvYXQsIHN0ZDo6X18yOjphbGxvY2F0b3I8ZmxvYXQ+PiBobnN3bGliOjpIaWVyYXJjaGljYWxOU1c8ZmxvYXQ+OjpnZXREYXRhQnlMYWJlbDxmbG9hdD4odW5zaWduZWQgbG9uZykgY29uc3SXATpobnN3bGliOjpIaWVyYXJjaGljYWxOU1c8ZmxvYXQ+OjptYXJrRGVsZXRlKHVuc2lnbmVkIGxvbmcpmAE8aG5zd2xpYjo6SGllcmFyY2hpY2FsTlNXPGZsb2F0Pjo6dW5tYXJrRGVsZXRlKHVuc2lnbmVkIGxvbmcpmQFtc3RkOjpfXzI6OnZlY3RvcjxmbG9hdCwgc3RkOjpfXzI6OmFsbG9jYXRvcjxmbG9hdD4+IGVtc2NyaXB0ZW46OnZlY0Zyb21KU0FycmF5PGZsb2F0PihlbXNjcmlwdGVuOjp2YWwgY29uc3QmKZoBpAFlbXNjcmlwdGVuOjppbnRlcm5hbDo6QmluZGluZ1R5cGU8c3RkOjpfXzI6OnZlY3RvcjxmbG9hdCwgc3RkOjpfXzI6OmFsbG9jYXRvcjxmbG9hdD4+LCB2b2lkPjo6dG9XaXJlVHlwZShzdGQ6Ol9fMjo6dmVjdG9yPGZsb2F0LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPGZsb2F0Pj4gY29uc3QmKZsBL3N0ZDo6X190aHJvd19iYWRfYXJyYXlfbmV3X2xlbmd0aFthYmk6djE1MDA3XSgpnAE3c3RkOjpfXzI6Ol9fdGhyb3dfbGVuZ3RoX2Vycm9yW2FiaTp2MTUwMDddKGNoYXIgY29uc3QqKZ0BKmVtc2NyaXB0ZW46OkwyU3BhY2U6OkwyU3BhY2UodW5zaWduZWQgaW50KZ4BNWhuc3dsaWI6OkwyU3FyKHZvaWQgY29uc3QqLCB2b2lkIGNvbnN0Kiwgdm9pZCBjb25zdCopnwEhaG5zd2xpYjo6TDJTcGFjZTo6Z2V0X2RhdGFfc2l6ZSgpoAEhaG5zd2xpYjo6TDJTcGFjZTo6Z2V0X2Rpc3RfZnVuYygpoQEnaG5zd2xpYjo6TDJTcGFjZTo6Z2V0X2Rpc3RfZnVuY19wYXJhbSgpogEcaG5zd2xpYjo6TDJTcGFjZTo6fkwyU3BhY2UoKaMBRGhuc3dsaWI6OklubmVyUHJvZHVjdERpc3RhbmNlKHZvaWQgY29uc3QqLCB2b2lkIGNvbnN0Kiwgdm9pZCBjb25zdCoppAExaG5zd2xpYjo6U3BhY2VJbnRlcmZhY2U8ZmxvYXQ+Ojp+U3BhY2VJbnRlcmZhY2UoKaUBOmVtc2NyaXB0ZW46OkN1c3RvbUZpbHRlckZ1bmN0b3I6Om9wZXJhdG9yKCkodW5zaWduZWQgbG9uZymmAUxobnN3bGliOjpCcnV0ZWZvcmNlU2VhcmNoPGZsb2F0Pjo6YWRkUG9pbnQodm9pZCBjb25zdCosIHVuc2lnbmVkIGxvbmcsIGJvb2wppwFqaG5zd2xpYjo6QnJ1dGVmb3JjZVNlYXJjaDxmbG9hdD46OnNlYXJjaEtubih2b2lkIGNvbnN0KiwgdW5zaWduZWQgbG9uZywgaG5zd2xpYjo6QmFzZUZpbHRlckZ1bmN0b3IqKSBjb25zdKgBlQJzdGQ6Ol9fMjo6cHJpb3JpdHlfcXVldWU8c3RkOjpfXzI6OnBhaXI8ZmxvYXQsIHVuc2lnbmVkIGxvbmc+LCBzdGQ6Ol9fMjo6dmVjdG9yPHN0ZDo6X18yOjpwYWlyPGZsb2F0LCB1bnNpZ25lZCBsb25nPiwgc3RkOjpfXzI6OmFsbG9jYXRvcjxzdGQ6Ol9fMjo6cGFpcjxmbG9hdCwgdW5zaWduZWQgbG9uZz4+Piwgc3RkOjpfXzI6Omxlc3M8c3RkOjpfXzI6OnBhaXI8ZmxvYXQsIHVuc2lnbmVkIGxvbmc+Pj46OnB1c2goc3RkOjpfXzI6OnBhaXI8ZmxvYXQsIHVuc2lnbmVkIGxvbmc+JiYpqQF3aG5zd2xpYjo6QWxnb3JpdGhtSW50ZXJmYWNlPGZsb2F0Pjo6c2VhcmNoS25uQ2xvc2VyRmlyc3Qodm9pZCBjb25zdCosIHVuc2lnbmVkIGxvbmcsIGhuc3dsaWI6OkJhc2VGaWx0ZXJGdW5jdG9yKikgY29uc3SqAYgBaG5zd2xpYjo6QnJ1dGVmb3JjZVNlYXJjaDxmbG9hdD46OnNhdmVJbmRleChzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPiwgc3RkOjpfXzI6OmFsbG9jYXRvcjxjaGFyPj4gY29uc3QmKasBtgFzdGQ6Ol9fMjo6YmFzaWNfb2ZzdHJlYW08Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+Pjo6YmFzaWNfb2ZzdHJlYW0oc3RkOjpfXzI6OmJhc2ljX3N0cmluZzxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4sIHN0ZDo6X18yOjphbGxvY2F0b3I8Y2hhcj4+IGNvbnN0JiwgdW5zaWduZWQgaW50KawBTnN0ZDo6X18yOjpiYXNpY19vZnN0cmVhbTxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+Ojp+YmFzaWNfb2ZzdHJlYW0oKa0BNWhuc3dsaWI6OkJydXRlZm9yY2VTZWFyY2g8ZmxvYXQ+Ojp+QnJ1dGVmb3JjZVNlYXJjaCgprgE3aG5zd2xpYjo6QnJ1dGVmb3JjZVNlYXJjaDxmbG9hdD46On5CcnV0ZWZvcmNlU2VhcmNoKCkuMa8BrwR2b2lkIHN0ZDo6X18yOjpfX2hhc2hfdGFibGU8c3RkOjpfXzI6Ol9faGFzaF92YWx1ZV90eXBlPHVuc2lnbmVkIGxvbmcsIHVuc2lnbmVkIGxvbmc+LCBzdGQ6Ol9fMjo6X191bm9yZGVyZWRfbWFwX2hhc2hlcjx1bnNpZ25lZCBsb25nLCBzdGQ6Ol9fMjo6X19oYXNoX3ZhbHVlX3R5cGU8dW5zaWduZWQgbG9uZywgdW5zaWduZWQgbG9uZz4sIHN0ZDo6X18yOjpoYXNoPHVuc2lnbmVkIGxvbmc+LCBzdGQ6Ol9fMjo6ZXF1YWxfdG88dW5zaWduZWQgbG9uZz4sIHRydWU+LCBzdGQ6Ol9fMjo6X191bm9yZGVyZWRfbWFwX2VxdWFsPHVuc2lnbmVkIGxvbmcsIHN0ZDo6X18yOjpfX2hhc2hfdmFsdWVfdHlwZTx1bnNpZ25lZCBsb25nLCB1bnNpZ25lZCBsb25nPiwgc3RkOjpfXzI6OmVxdWFsX3RvPHVuc2lnbmVkIGxvbmc+LCBzdGQ6Ol9fMjo6aGFzaDx1bnNpZ25lZCBsb25nPiwgdHJ1ZT4sIHN0ZDo6X18yOjphbGxvY2F0b3I8c3RkOjpfXzI6Ol9faGFzaF92YWx1ZV90eXBlPHVuc2lnbmVkIGxvbmcsIHVuc2lnbmVkIGxvbmc+Pj46Ol9fZG9fcmVoYXNoPHRydWU+KHVuc2lnbmVkIGxvbmcpsAG2AXN0ZDo6X18yOjpiYXNpY19pZnN0cmVhbTxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+OjpiYXNpY19pZnN0cmVhbShzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPiwgc3RkOjpfXzI6OmFsbG9jYXRvcjxjaGFyPj4gY29uc3QmLCB1bnNpZ25lZCBpbnQpsQFOc3RkOjpfXzI6OmJhc2ljX2lmc3RyZWFtPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPj46On5iYXNpY19pZnN0cmVhbSgpsgH0AWVtc2NyaXB0ZW46OkhpZXJhcmNoaWNhbE5TVzo6SGllcmFyY2hpY2FsTlNXKHN0ZDo6X18yOjpiYXNpY19zdHJpbmc8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPGNoYXI+PiBjb25zdCYsIHVuc2lnbmVkIGludCwgc3RkOjpfXzI6OmJhc2ljX3N0cmluZzxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4sIHN0ZDo6X18yOjphbGxvY2F0b3I8Y2hhcj4+IGNvbnN0JimzAZABc3RkOjpfXzI6Ol9fdHJhbnNhY3Rpb248c3RkOjpfXzI6OnZlY3RvcjxzdGQ6Ol9fMjo6bXV0ZXgsIHN0ZDo6X18yOjphbGxvY2F0b3I8c3RkOjpfXzI6Om11dGV4Pj46Ol9fZGVzdHJveV92ZWN0b3I+Ojp+X190cmFuc2FjdGlvblthYmk6djE1MDA3XSgptAEzaG5zd2xpYjo6VmlzaXRlZExpc3RQb29sOjpWaXNpdGVkTGlzdFBvb2woaW50LCBpbnQptQFec3RkOjpfXzI6OnZlY3RvcjxzdGQ6Ol9fMjo6bXV0ZXgsIHN0ZDo6X18yOjphbGxvY2F0b3I8c3RkOjpfXzI6Om11dGV4Pj46On52ZWN0b3JbYWJpOnYxNTAwN10oKbYBanN0ZDo6X18yOjpkZXF1ZTxobnN3bGliOjpWaXNpdGVkTGlzdCosIHN0ZDo6X18yOjphbGxvY2F0b3I8aG5zd2xpYjo6VmlzaXRlZExpc3QqPj46Ol9fYWRkX2Zyb250X2NhcGFjaXR5KCm3AWpzdGQ6Ol9fMjo6X19kZXF1ZV9iYXNlPGhuc3dsaWI6OlZpc2l0ZWRMaXN0Kiwgc3RkOjpfXzI6OmFsbG9jYXRvcjxobnN3bGliOjpWaXNpdGVkTGlzdCo+Pjo6fl9fZGVxdWVfYmFzZSgpuAFLaG5zd2xpYjo6SGllcmFyY2hpY2FsTlNXPGZsb2F0Pjo6YWRkUG9pbnQodm9pZCBjb25zdCosIHVuc2lnbmVkIGxvbmcsIGJvb2wpuQFKaG5zd2xpYjo6SGllcmFyY2hpY2FsTlNXPGZsb2F0Pjo6YWRkUG9pbnQodm9pZCBjb25zdCosIHVuc2lnbmVkIGxvbmcsIGludCm6AURobnN3bGliOjpIaWVyYXJjaGljYWxOU1c8ZmxvYXQ+Ojp1bm1hcmtEZWxldGVkSW50ZXJuYWwodW5zaWduZWQgaW50KbsBTmhuc3dsaWI6OkhpZXJhcmNoaWNhbE5TVzxmbG9hdD46OnVwZGF0ZVBvaW50KHZvaWQgY29uc3QqLCB1bnNpZ25lZCBpbnQsIGZsb2F0KbwBaWhuc3dsaWI6OkhpZXJhcmNoaWNhbE5TVzxmbG9hdD46OnNlYXJjaEtubih2b2lkIGNvbnN0KiwgdW5zaWduZWQgbG9uZywgaG5zd2xpYjo6QmFzZUZpbHRlckZ1bmN0b3IqKSBjb25zdL0BhwFobnN3bGliOjpIaWVyYXJjaGljYWxOU1c8ZmxvYXQ+OjpzYXZlSW5kZXgoc3RkOjpfXzI6OmJhc2ljX3N0cmluZzxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4sIHN0ZDo6X18yOjphbGxvY2F0b3I8Y2hhcj4+IGNvbnN0Jim+ATNobnN3bGliOjpIaWVyYXJjaGljYWxOU1c8ZmxvYXQ+Ojp+SGllcmFyY2hpY2FsTlNXKCm/ASxobnN3bGliOjpWaXNpdGVkTGlzdFBvb2w6On5WaXNpdGVkTGlzdFBvb2woKcABNWhuc3dsaWI6OkhpZXJhcmNoaWNhbE5TVzxmbG9hdD46On5IaWVyYXJjaGljYWxOU1coKS4xwQGIAXN0ZDo6X18yOjpfX3NwbGl0X2J1ZmZlcjxobnN3bGliOjpWaXNpdGVkTGlzdCoqLCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPGhuc3dsaWI6OlZpc2l0ZWRMaXN0Kio+Pjo6cHVzaF9mcm9udChobnN3bGliOjpWaXNpdGVkTGlzdCoqIGNvbnN0JinCAVBobnN3bGliOjpIaWVyYXJjaGljYWxOU1c8ZmxvYXQ+OjpzZWFyY2hCYXNlTGF5ZXIodW5zaWduZWQgaW50LCB2b2lkIGNvbnN0KiwgaW50KcMBmwJ2b2lkIHN0ZDo6X18yOjpwcmlvcml0eV9xdWV1ZTxzdGQ6Ol9fMjo6cGFpcjxmbG9hdCwgdW5zaWduZWQgaW50Piwgc3RkOjpfXzI6OnZlY3RvcjxzdGQ6Ol9fMjo6cGFpcjxmbG9hdCwgdW5zaWduZWQgaW50Piwgc3RkOjpfXzI6OmFsbG9jYXRvcjxzdGQ6Ol9fMjo6cGFpcjxmbG9hdCwgdW5zaWduZWQgaW50Pj4+LCBobnN3bGliOjpIaWVyYXJjaGljYWxOU1c8ZmxvYXQ+OjpDb21wYXJlQnlGaXJzdD46OmVtcGxhY2U8ZmxvYXQsIHVuc2lnbmVkIGludCY+KGZsb2F0JiYsIHVuc2lnbmVkIGludCYpxAHCAmhuc3dsaWI6OkhpZXJhcmNoaWNhbE5TVzxmbG9hdD46Om11dHVhbGx5Q29ubmVjdE5ld0VsZW1lbnQodm9pZCBjb25zdCosIHVuc2lnbmVkIGludCwgc3RkOjpfXzI6OnByaW9yaXR5X3F1ZXVlPHN0ZDo6X18yOjpwYWlyPGZsb2F0LCB1bnNpZ25lZCBpbnQ+LCBzdGQ6Ol9fMjo6dmVjdG9yPHN0ZDo6X18yOjpwYWlyPGZsb2F0LCB1bnNpZ25lZCBpbnQ+LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPHN0ZDo6X18yOjpwYWlyPGZsb2F0LCB1bnNpZ25lZCBpbnQ+Pj4sIGhuc3dsaWI6OkhpZXJhcmNoaWNhbE5TVzxmbG9hdD46OkNvbXBhcmVCeUZpcnN0PiYsIGludCwgYm9vbCnFAW5obnN3bGliOjpIaWVyYXJjaGljYWxOU1c8ZmxvYXQ+OjpyZXBhaXJDb25uZWN0aW9uc0ZvclVwZGF0ZSh2b2lkIGNvbnN0KiwgdW5zaWduZWQgaW50LCB1bnNpZ25lZCBpbnQsIGludCwgaW50KcYBzAJzdGQ6Ol9fMjo6cGFpcjxzdGQ6Ol9fMjo6X19oYXNoX2l0ZXJhdG9yPHN0ZDo6X18yOjpfX2hhc2hfbm9kZTx1bnNpZ25lZCBpbnQsIHZvaWQqPio+LCBib29sPiBzdGQ6Ol9fMjo6X19oYXNoX3RhYmxlPHVuc2lnbmVkIGludCwgc3RkOjpfXzI6Omhhc2g8dW5zaWduZWQgaW50Piwgc3RkOjpfXzI6OmVxdWFsX3RvPHVuc2lnbmVkIGludD4sIHN0ZDo6X18yOjphbGxvY2F0b3I8dW5zaWduZWQgaW50Pj46Ol9fZW1wbGFjZV91bmlxdWVfa2V5X2FyZ3M8dW5zaWduZWQgaW50LCB1bnNpZ25lZCBpbnQgY29uc3QmPih1bnNpZ25lZCBpbnQgY29uc3QmLCB1bnNpZ25lZCBpbnQgY29uc3QmKccBqgJobnN3bGliOjpIaWVyYXJjaGljYWxOU1c8ZmxvYXQ+OjpnZXROZWlnaGJvcnNCeUhldXJpc3RpYzIoc3RkOjpfXzI6OnByaW9yaXR5X3F1ZXVlPHN0ZDo6X18yOjpwYWlyPGZsb2F0LCB1bnNpZ25lZCBpbnQ+LCBzdGQ6Ol9fMjo6dmVjdG9yPHN0ZDo6X18yOjpwYWlyPGZsb2F0LCB1bnNpZ25lZCBpbnQ+LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPHN0ZDo6X18yOjpwYWlyPGZsb2F0LCB1bnNpZ25lZCBpbnQ+Pj4sIGhuc3dsaWI6OkhpZXJhcmNoaWNhbE5TVzxmbG9hdD46OkNvbXBhcmVCeUZpcnN0PiYsIHVuc2lnbmVkIGxvbmcpyAEuaG5zd2xpYjo6VmlzaXRlZExpc3RQb29sOjpnZXRGcmVlVmlzaXRlZExpc3QoKckBtwFobnN3bGliOjpIaWVyYXJjaGljYWxOU1c8ZmxvYXQ+Ojpsb2FkSW5kZXgoc3RkOjpfXzI6OmJhc2ljX3N0cmluZzxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4sIHN0ZDo6X18yOjphbGxvY2F0b3I8Y2hhcj4+IGNvbnN0JiwgaG5zd2xpYjo6U3BhY2VJbnRlcmZhY2U8ZmxvYXQ+KiwgdW5zaWduZWQgbG9uZynKAfcBc3RkOjpfXzI6OnZlY3RvcjxzdGQ6Ol9fMjo6dmVjdG9yPGZsb2F0LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPGZsb2F0Pj4sIHN0ZDo6X18yOjphbGxvY2F0b3I8c3RkOjpfXzI6OnZlY3RvcjxmbG9hdCwgc3RkOjpfXzI6OmFsbG9jYXRvcjxmbG9hdD4+Pj4gZW1zY3JpcHRlbjo6dmVjRnJvbUpTQXJyYXk8c3RkOjpfXzI6OnZlY3RvcjxmbG9hdCwgc3RkOjpfXzI6OmFsbG9jYXRvcjxmbG9hdD4+PihlbXNjcmlwdGVuOjp2YWwgY29uc3QmKcsBggFzdGQ6Ol9fMjo6dmVjdG9yPHVuc2lnbmVkIGludCwgc3RkOjpfXzI6OmFsbG9jYXRvcjx1bnNpZ25lZCBpbnQ+PiBlbXNjcmlwdGVuOjp2ZWNGcm9tSlNBcnJheTx1bnNpZ25lZCBpbnQ+KGVtc2NyaXB0ZW46OnZhbCBjb25zdCYpzAGmAXN0ZDo6X18yOjp2ZWN0b3I8c3RkOjpfXzI6OnZlY3RvcjxmbG9hdCwgc3RkOjpfXzI6OmFsbG9jYXRvcjxmbG9hdD4+LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPHN0ZDo6X18yOjp2ZWN0b3I8ZmxvYXQsIHN0ZDo6X18yOjphbGxvY2F0b3I8ZmxvYXQ+Pj4+Ojp+dmVjdG9yW2FiaTp2MTUwMDddKCnNAcABZW1zY3JpcHRlbjo6aW50ZXJuYWw6OkJpbmRpbmdUeXBlPHN0ZDo6X18yOjp2ZWN0b3I8dW5zaWduZWQgaW50LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPHVuc2lnbmVkIGludD4+LCB2b2lkPjo6dG9XaXJlVHlwZShzdGQ6Ol9fMjo6dmVjdG9yPHVuc2lnbmVkIGludCwgc3RkOjpfXzI6OmFsbG9jYXRvcjx1bnNpZ25lZCBpbnQ+PiBjb25zdCYpzgFCaG5zd2xpYjo6SGllcmFyY2hpY2FsTlNXPGZsb2F0Pjo6bWFya0RlbGV0ZWRJbnRlcm5hbCh1bnNpZ25lZCBpbnQpzwENX19nZXRUeXBlTmFtZdABG19lbWJpbmRfaW5pdGlhbGl6ZV9iaW5kaW5nc9EBFWVtYmluZF9pbml0X2J1aWx0aW4oKdIBCF9fbWVtY3B50wEHbWVtbW92ZdQBBm1lbXNldNUBA2xvZ9YBBm1lbWNoctcBBm1lbWNtcNgBB2lwcmludGbZAQpfX2xvY2tmaWxl2gEVZW1zY3JpcHRlbl9mdXRleF93YWtl2wEUX19wdGhyZWFkX211dGV4X2xvY2vcAQlfX3Rvd3JpdGXdAQlfX2Z3cml0ZXjeAQZmd3JpdGXfAQpfX292ZXJmbG934AEEcHV0c+EBDV9fc3RkaW9fd3JpdGXiARhfX2Vtc2NyaXB0ZW5fc3Rkb3V0X3NlZWvjAQZzdHJsZW7kARBfX2Vycm5vX2xvY2F0aW9u5QEFZnJleHDmARNfX3ZmcHJpbnRmX2ludGVybmFs5wELcHJpbnRmX2NvcmXoAQNvdXTpAQZnZXRpbnTqAQdwb3BfYXJn6wEFZm10X3XsAQNwYWTtAQh2ZnByaW50Zu4BBmZtdF9mcO8BE3BvcF9hcmdfbG9uZ19kb3VibGXwAQd3Y3J0b21i8QEGd2N0b21i8gEIZGxtYWxsb2PzAQZkbGZyZWX0AQlkbHJlYWxsb2P1AQ1kaXNwb3NlX2NodW5r9gEEc2Jya/cBCV9fYXNobHRpM/gBCV9fbHNocnRpM/kBDF9fdHJ1bmN0ZmRmMvoBJXN0ZDo6X18yOjpfX25leHRfcHJpbWUodW5zaWduZWQgbG9uZyn7AZkBdW5zaWduZWQgaW50IGNvbnN0KiBzdGQ6Ol9fMjo6bG93ZXJfYm91bmRbYWJpOnYxNTAwN108dW5zaWduZWQgaW50IGNvbnN0KiwgdW5zaWduZWQgbG9uZz4odW5zaWduZWQgaW50IGNvbnN0KiwgdW5zaWduZWQgaW50IGNvbnN0KiwgdW5zaWduZWQgbG9uZyBjb25zdCYp/AE5c3RkOjpfXzI6Ol9fdGhyb3dfb3ZlcmZsb3dfZXJyb3JbYWJpOnYxNTAwN10oY2hhciBjb25zdCop/QFkdW5zaWduZWQgaW50IGNvbnN0JiBzdGQ6Ol9fMjo6X19pZGVudGl0eTo6b3BlcmF0b3IoKTx1bnNpZ25lZCBpbnQgY29uc3QmPih1bnNpZ25lZCBpbnQgY29uc3QmKSBjb25zdP4BC19fc3RyY2hybnVs/wEGc3RyY2hygAIMX19zdGRpb19zZWVrgQIMX19zdGRpb19yZWFkggINX19zdGRpb19jbG9zZYMCEV9fZnNlZWtvX3VubG9ja2VkhAIIX19mc2Vla2+FAgZmZmx1c2iGAgZmY2xvc2WHAghfX3RvcmVhZIgCBWZyZWFkiQIRX19mdGVsbG9fdW5sb2NrZWSKAkRzdGQ6Ol9fMjo6YmFzaWNfaW9zPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPj46On5iYXNpY19pb3MoKYsCUHN0ZDo6X18yOjpiYXNpY19zdHJlYW1idWY8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+Pjo6fmJhc2ljX3N0cmVhbWJ1ZigpjAJSc3RkOjpfXzI6OmJhc2ljX3N0cmVhbWJ1ZjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+Ojp+YmFzaWNfc3RyZWFtYnVmKCkuMY0CXHN0ZDo6X18yOjpiYXNpY19zdHJlYW1idWY8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+Pjo6aW1idWUoc3RkOjpfXzI6OmxvY2FsZSBjb25zdCYpjgJRc3RkOjpfXzI6OmJhc2ljX3N0cmVhbWJ1ZjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+OjpzZXRidWYoY2hhciosIGxvbmcpjwJ7c3RkOjpfXzI6OmJhc2ljX3N0cmVhbWJ1ZjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+OjpzZWVrb2ZmKGxvbmcgbG9uZywgc3RkOjpfXzI6Omlvc19iYXNlOjpzZWVrZGlyLCB1bnNpZ25lZCBpbnQpkAJwc3RkOjpfXzI6OmJhc2ljX3N0cmVhbWJ1ZjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+OjpzZWVrcG9zKHN0ZDo6X18yOjpmcG9zPF9fbWJzdGF0ZV90PiwgdW5zaWduZWQgaW50KZECUXN0ZDo6X18yOjpiYXNpY19zdHJlYW1idWY8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+Pjo6eHNnZXRuKGNoYXIqLCBsb25nKZICRHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPjo6Y29weShjaGFyKiwgY2hhciBjb25zdCosIHVuc2lnbmVkIGxvbmcpkwK6AXN0ZDo6X18yOjplbmFibGVfaWY8X19pc19jcHAxN19yYW5kb21fYWNjZXNzX2l0ZXJhdG9yPGNoYXIgY29uc3QqPjo6dmFsdWUsIGNoYXIqPjo6dHlwZSBzdGQ6Ol9fMjo6Y29weV9uW2FiaTp2MTUwMDddPGNoYXIgY29uc3QqLCB1bnNpZ25lZCBsb25nLCBjaGFyKj4oY2hhciBjb25zdCosIHVuc2lnbmVkIGxvbmcsIGNoYXIqKZQCSXN0ZDo6X18yOjpiYXNpY19zdHJlYW1idWY8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+Pjo6dW5kZXJmbG93KCmVAkVzdGQ6Ol9fMjo6YmFzaWNfc3RyZWFtYnVmPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPj46OnVmbG93KCmWAkxzdGQ6Ol9fMjo6YmFzaWNfc3RyZWFtYnVmPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPj46OnBiYWNrZmFpbChpbnQplwJXc3RkOjpfXzI6OmJhc2ljX3N0cmVhbWJ1ZjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+Ojp4c3B1dG4oY2hhciBjb25zdCosIGxvbmcpmAJMc3RkOjpfXzI6OmJhc2ljX2lzdHJlYW08Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+Pjo6fmJhc2ljX2lzdHJlYW0oKZkCTnN0ZDo6X18yOjpiYXNpY19pc3RyZWFtPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPj46On5iYXNpY19pc3RyZWFtKCkuMZoCXXZpcnR1YWwgdGh1bmsgdG8gc3RkOjpfXzI6OmJhc2ljX2lzdHJlYW08Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+Pjo6fmJhc2ljX2lzdHJlYW0oKZsCTnN0ZDo6X18yOjpiYXNpY19pc3RyZWFtPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPj46On5iYXNpY19pc3RyZWFtKCkuMpwCX3ZpcnR1YWwgdGh1bmsgdG8gc3RkOjpfXzI6OmJhc2ljX2lzdHJlYW08Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+Pjo6fmJhc2ljX2lzdHJlYW0oKS4xnQKNAXN0ZDo6X18yOjpiYXNpY19pc3RyZWFtPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPj46OnNlbnRyeTo6c2VudHJ5KHN0ZDo6X18yOjpiYXNpY19pc3RyZWFtPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPj4mLCBib29sKZ4CQ3N0ZDo6X18yOjpiYXNpY19vc3RyZWFtPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPj46OmZsdXNoKCmfAmxzdGQ6Ol9fMjo6Y3R5cGU8Y2hhcj4gY29uc3QmIHN0ZDo6X18yOjp1c2VfZmFjZXRbYWJpOnYxNTAwN108c3RkOjpfXzI6OmN0eXBlPGNoYXI+PihzdGQ6Ol9fMjo6bG9jYWxlIGNvbnN0JimgAlpzdGQ6Ol9fMjo6YmFzaWNfaW9zPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPj46OnNldHN0YXRlW2FiaTp2MTUwMDddKHVuc2lnbmVkIGludCmhAtoBYm9vbCBzdGQ6Ol9fMjo6b3BlcmF0b3I9PVthYmk6djE1MDA3XTxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+KHN0ZDo6X18yOjppc3RyZWFtYnVmX2l0ZXJhdG9yPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPj4gY29uc3QmLCBzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+IGNvbnN0JimiAlpzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+OjpvcGVyYXRvcisrW2FiaTp2MTUwMDddKCmjAlJzdGQ6Ol9fMjo6YmFzaWNfc3RyZWFtYnVmPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPj46OnNidW1wY1thYmk6djE1MDA3XSgppAJNc3RkOjpfXzI6OmJhc2ljX2lzdHJlYW08Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+Pjo6cmVhZChjaGFyKiwgbG9uZymlAkNzdGQ6Ol9fMjo6YmFzaWNfaXN0cmVhbTxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+Ojp0ZWxsZygppgJpc3RkOjpfXzI6OmJhc2ljX2lzdHJlYW08Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+Pjo6c2Vla2cobG9uZyBsb25nLCBzdGQ6Ol9fMjo6aW9zX2Jhc2U6OnNlZWtkaXIppwJOc3RkOjpfXzI6OmJhc2ljX29zdHJlYW08Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+Pjo6fmJhc2ljX29zdHJlYW0oKS4xqAJddmlydHVhbCB0aHVuayB0byBzdGQ6Ol9fMjo6YmFzaWNfb3N0cmVhbTxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+Ojp+YmFzaWNfb3N0cmVhbSgpqQJOc3RkOjpfXzI6OmJhc2ljX29zdHJlYW08Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+Pjo6fmJhc2ljX29zdHJlYW0oKS4yqgJfdmlydHVhbCB0aHVuayB0byBzdGQ6Ol9fMjo6YmFzaWNfb3N0cmVhbTxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+Ojp+YmFzaWNfb3N0cmVhbSgpLjGrAocBc3RkOjpfXzI6OmJhc2ljX29zdHJlYW08Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+Pjo6c2VudHJ5OjpzZW50cnkoc3RkOjpfXzI6OmJhc2ljX29zdHJlYW08Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+PiYprAJNc3RkOjpfXzI6OmJhc2ljX29zdHJlYW08Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+Pjo6c2VudHJ5Ojp+c2VudHJ5KCmtAl1zdGQ6Ol9fMjo6b3N0cmVhbWJ1Zl9pdGVyYXRvcjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+OjpvcGVyYXRvcj1bYWJpOnYxNTAwN10oY2hhcimuAlRzdGQ6Ol9fMjo6YmFzaWNfb3N0cmVhbTxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+Ojp3cml0ZShjaGFyIGNvbnN0KiwgbG9uZymvAk1zdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8d2NoYXJfdD46OmNvcHkod2NoYXJfdCosIHdjaGFyX3QgY29uc3QqLCB1bnNpZ25lZCBsb25nKbACcnN0ZDo6X18yOjpjdHlwZTx3Y2hhcl90PiBjb25zdCYgc3RkOjpfXzI6OnVzZV9mYWNldFthYmk6djE1MDA3XTxzdGQ6Ol9fMjo6Y3R5cGU8d2NoYXJfdD4+KHN0ZDo6X18yOjpsb2NhbGUgY29uc3QmKbEC7AFib29sIHN0ZDo6X18yOjpvcGVyYXRvcj09W2FiaTp2MTUwMDddPHdjaGFyX3QsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czx3Y2hhcl90Pj4oc3RkOjpfXzI6OmlzdHJlYW1idWZfaXRlcmF0b3I8d2NoYXJfdCwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPHdjaGFyX3Q+PiBjb25zdCYsIHN0ZDo6X18yOjppc3RyZWFtYnVmX2l0ZXJhdG9yPHdjaGFyX3QsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czx3Y2hhcl90Pj4gY29uc3QmKbICYHN0ZDo6X18yOjppc3RyZWFtYnVmX2l0ZXJhdG9yPHdjaGFyX3QsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czx3Y2hhcl90Pj46Om9wZXJhdG9yKytbYWJpOnYxNTAwN10oKbMCWHN0ZDo6X18yOjpiYXNpY19zdHJlYW1idWY8d2NoYXJfdCwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPHdjaGFyX3Q+Pjo6c2J1bXBjW2FiaTp2MTUwMDddKCm0AmZzdGQ6Ol9fMjo6b3N0cmVhbWJ1Zl9pdGVyYXRvcjx3Y2hhcl90LCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8d2NoYXJfdD4+OjpvcGVyYXRvcj1bYWJpOnYxNTAwN10od2NoYXJfdCm1AlZzdGQ6Ol9fMjo6YmFzaWNfc3RyZWFtYnVmPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPj46OmdwdHJbYWJpOnYxNTAwN10oKSBjb25zdLYCwwFzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPiwgc3RkOjpfXzI6OmFsbG9jYXRvcjxjaGFyPj46Om9wZXJhdG9yPVthYmk6djE1MDA3XShzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPiwgc3RkOjpfXzI6OmFsbG9jYXRvcjxjaGFyPj4mJim3ArwBc3RkOjpfXzI6OmVuYWJsZV9pZjxfX2lzX2NwcDE3X2ZvcndhcmRfaXRlcmF0b3I8Y2hhcio+Ojp2YWx1ZSwgdm9pZD46OnR5cGUgc3RkOjpfXzI6OmJhc2ljX3N0cmluZzxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4sIHN0ZDo6X18yOjphbGxvY2F0b3I8Y2hhcj4+OjpfX2luaXQ8Y2hhcio+KGNoYXIqLCBjaGFyKim4AndzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPiwgc3RkOjpfXzI6OmFsbG9jYXRvcjxjaGFyPj46OnJlc2l6ZVthYmk6djE1MDA3XSh1bnNpZ25lZCBsb25nKbkCW3N0ZDo6X18yOjpiYXNpY19maWxlYnVmPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPj46Om9wZW4oY2hhciBjb25zdCosIHVuc2lnbmVkIGludCm6AktzdGQ6Ol9fMjo6YmFzaWNfZmlsZWJ1ZjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+OjpiYXNpY19maWxlYnVmKCm7ApYBc3RkOjpfXzI6OmNvZGVjdnQ8Y2hhciwgY2hhciwgX19tYnN0YXRlX3Q+IGNvbnN0JiBzdGQ6Ol9fMjo6dXNlX2ZhY2V0W2FiaTp2MTUwMDddPHN0ZDo6X18yOjpjb2RlY3Z0PGNoYXIsIGNoYXIsIF9fbWJzdGF0ZV90Pj4oc3RkOjpfXzI6OmxvY2FsZSBjb25zdCYpvAJMc3RkOjpfXzI6OmJhc2ljX2ZpbGVidWY8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+Pjo6fmJhc2ljX2ZpbGVidWYoKb0CQ3N0ZDo6X18yOjpiYXNpY19maWxlYnVmPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPj46OmNsb3NlKCm+AtQBc3RkOjpfXzI6OnVuaXF1ZV9wdHI8X0lPX0ZJTEUsIGludCAoKikoX0lPX0ZJTEUqKT46OnVuaXF1ZV9wdHJbYWJpOnYxNTAwN108dHJ1ZSwgdm9pZD4oX0lPX0ZJTEUqLCBzdGQ6Ol9fMjo6X19kZXBlbmRlbnRfdHlwZTxzdGQ6Ol9fMjo6X191bmlxdWVfcHRyX2RlbGV0ZXJfc2ZpbmFlPGludCAoKikoX0lPX0ZJTEUqKT4sIHRydWU+OjpfX2dvb2RfcnZhbF9yZWZfdHlwZSm/Ak5zdGQ6Ol9fMjo6YmFzaWNfZmlsZWJ1ZjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+Ojp+YmFzaWNfZmlsZWJ1ZigpLjHAAlBzdGQ6Ol9fMjo6dW5pcXVlX3B0cjxfSU9fRklMRSwgaW50ICgqKShfSU9fRklMRSopPjo6cmVzZXRbYWJpOnYxNTAwN10oX0lPX0ZJTEUqKcECR3N0ZDo6X18yOjpiYXNpY19maWxlYnVmPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPj46OnVuZGVyZmxvdygpwgIoc3RkOjpfXzI6Ol9fdGhyb3dfYmFkX2Nhc3RbYWJpOnYxNTAwN10oKcMCSnN0ZDo6X18yOjpiYXNpY19maWxlYnVmPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPj46OnBiYWNrZmFpbChpbnQpxAJJc3RkOjpfXzI6OmJhc2ljX2ZpbGVidWY8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+Pjo6b3ZlcmZsb3coaW50KcUCT3N0ZDo6X18yOjpiYXNpY19maWxlYnVmPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPj46OnNldGJ1ZihjaGFyKiwgbG9uZynGAnlzdGQ6Ol9fMjo6YmFzaWNfZmlsZWJ1ZjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+OjpzZWVrb2ZmKGxvbmcgbG9uZywgc3RkOjpfXzI6Omlvc19iYXNlOjpzZWVrZGlyLCB1bnNpZ25lZCBpbnQpxwJuc3RkOjpfXzI6OmJhc2ljX2ZpbGVidWY8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+Pjo6c2Vla3BvcyhzdGQ6Ol9fMjo6ZnBvczxfX21ic3RhdGVfdD4sIHVuc2lnbmVkIGludCnIAkJzdGQ6Ol9fMjo6YmFzaWNfZmlsZWJ1ZjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+OjpzeW5jKCnJAlpzdGQ6Ol9fMjo6YmFzaWNfZmlsZWJ1ZjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+OjppbWJ1ZShzdGQ6Ol9fMjo6bG9jYWxlIGNvbnN0JinKAlBzdGQ6Ol9fMjo6YmFzaWNfaWZzdHJlYW08Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+Pjo6fmJhc2ljX2lmc3RyZWFtKCkuMcsCX3ZpcnR1YWwgdGh1bmsgdG8gc3RkOjpfXzI6OmJhc2ljX2lmc3RyZWFtPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPj46On5iYXNpY19pZnN0cmVhbSgpzAJhdmlydHVhbCB0aHVuayB0byBzdGQ6Ol9fMjo6YmFzaWNfaWZzdHJlYW08Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+Pjo6fmJhc2ljX2lmc3RyZWFtKCkuMc0CUHN0ZDo6X18yOjpiYXNpY19vZnN0cmVhbTxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+Ojp+YmFzaWNfb2ZzdHJlYW0oKS4xzgJfdmlydHVhbCB0aHVuayB0byBzdGQ6Ol9fMjo6YmFzaWNfb2ZzdHJlYW08Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+Pjo6fmJhc2ljX29mc3RyZWFtKCnPAmF2aXJ0dWFsIHRodW5rIHRvIHN0ZDo6X18yOjpiYXNpY19vZnN0cmVhbTxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+Ojp+YmFzaWNfb2ZzdHJlYW0oKS4x0AJVY2hhciogc3RkOjpfXzI6OmNvcHlbYWJpOnYxNTAwN108Y2hhciBjb25zdCosIGNoYXIqPihjaGFyIGNvbnN0KiwgY2hhciBjb25zdCosIGNoYXIqKdECXWF1dG8gc3RkOjpfXzI6Ol9fdW53cmFwX3JhbmdlW2FiaTp2MTUwMDddPGNoYXIgY29uc3QqLCBjaGFyIGNvbnN0Kj4oY2hhciBjb25zdCosIGNoYXIgY29uc3QqKdICfXN0ZDo6X18yOjpwYWlyPGNoYXIgY29uc3QqLCBjaGFyKj4gc3RkOjpfXzI6Ol9fY29weV9pbXBsW2FiaTp2MTUwMDddPGNoYXIgY29uc3QsIGNoYXIsIHZvaWQ+KGNoYXIgY29uc3QqLCBjaGFyIGNvbnN0KiwgY2hhciop0wJ/c3RkOjpfXzI6OmFsbG9jYXRvcl90cmFpdHM8c3RkOjpfXzI6OmFsbG9jYXRvcjxjaGFyPj46OmRlYWxsb2NhdGVbYWJpOnYxNTAwN10oc3RkOjpfXzI6OmFsbG9jYXRvcjxjaGFyPiYsIGNoYXIqLCB1bnNpZ25lZCBsb25nKdQCTnN0ZDo6X18yOjpfX2xpYmNwcF9kZWFsbG9jYXRlW2FiaTp2MTUwMDddKHZvaWQqLCB1bnNpZ25lZCBsb25nLCB1bnNpZ25lZCBsb25nKdUCzQFzdGQ6Ol9fMjo6X19hbGxvY2F0aW9uX3Jlc3VsdDxzdGQ6Ol9fMjo6YWxsb2NhdG9yX3RyYWl0czxzdGQ6Ol9fMjo6YWxsb2NhdG9yPGNoYXI+Pjo6cG9pbnRlcj4gc3RkOjpfXzI6Ol9fYWxsb2NhdGVfYXRfbGVhc3RbYWJpOnYxNTAwN108c3RkOjpfXzI6OmFsbG9jYXRvcjxjaGFyPj4oc3RkOjpfXzI6OmFsbG9jYXRvcjxjaGFyPiYsIHVuc2lnbmVkIGxvbmcp1gI+c3RkOjpfXzI6OmFsbG9jYXRvcjxjaGFyPjo6YWxsb2NhdGVbYWJpOnYxNTAwN10odW5zaWduZWQgbG9uZynXAkVzdGQ6Ol9fMjo6X19saWJjcHBfYWxsb2NhdGVbYWJpOnYxNTAwN10odW5zaWduZWQgbG9uZywgdW5zaWduZWQgbG9uZynYAmRzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+OjpfX3Rlc3RfZm9yX2VvZlthYmk6djE1MDA3XSgpIGNvbnN02QJqc3RkOjpfXzI6OmlzdHJlYW1idWZfaXRlcmF0b3I8d2NoYXJfdCwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPHdjaGFyX3Q+Pjo6X190ZXN0X2Zvcl9lb2ZbYWJpOnYxNTAwN10oKSBjb25zdNoCK3N0ZDo6X18yOjpfX2lvc3RyZWFtX2NhdGVnb3J5OjpuYW1lKCkgY29uc3TbAjFzdGQ6Ol9fMjo6X19pb3N0cmVhbV9jYXRlZ29yeTo6bWVzc2FnZShpbnQpIGNvbnN03AInc3RkOjpfXzI6Omlvc19iYXNlOjpmYWlsdXJlOjp+ZmFpbHVyZSgp3QIpc3RkOjpfXzI6Omlvc19iYXNlOjpmYWlsdXJlOjp+ZmFpbHVyZSgpLjHeAidzdGQ6Ol9fMjo6aW9zX2Jhc2U6OmNsZWFyKHVuc2lnbmVkIGludCnfAh9zdGQ6Ol9fMjo6aW9zX2Jhc2U6On5pb3NfYmFzZSgp4AIhc3RkOjpfXzI6Omlvc19iYXNlOjp+aW9zX2Jhc2UoKS4x4QIfc3RkOjpfXzI6Omlvc19iYXNlOjppbml0KHZvaWQqKeICN3N0ZDo6X18yOjppb3NfYmFzZTo6X19zZXRfYmFkYml0X2FuZF9jb25zaWRlcl9yZXRocm93KCnjAgdfX3NobGlt5AIIX19zaGdldGPlAgtfX2Zsb2F0c2l0ZuYCCF9fbXVsdGYz5wIIX19hZGR0ZjPoAg1fX2V4dGVuZGRmdGYy6QIHX19sZXRmMuoCB19fZ2V0ZjLrAgZzY2FsYm7sAgljb3B5c2lnbmztAg1fX2Zsb2F0dW5zaXRm7gIIX19zdWJ0ZjPvAgdzY2FsYm5s8AIIX19tdWx0aTPxAghfX2RpdnRmM/ICBWZtb2Rs8wILX19mbG9hdHNjYW70AgdzY2FuZXhw9QIMX190cnVuY3Rmc2Yy9gIHbWJydG93Y/cCCXN0b3JlX2ludPgCB3Zzc2Nhbmb5AgtzdHJpbmdfcmVhZPoCBnN0cmNtcPsCBXN3YXBj/AIHc3RybmNtcP0CBmdldGVudv4CDF9fZ2V0X2xvY2FsZf8CCXZzbnByaW50ZoADCHNuX3dyaXRlgQMGc3NjYW5mggMIc25wcmludGaDAwpmcmVlbG9jYWxlhAMGc3RyY3B5hQMGd2NzbGVuhgMJbWJzcnRvd2NzhwMGc3RydG94iAMIc3RydG94LjGJA11zdGQ6Ol9fMjo6Y29sbGF0ZTxjaGFyPjo6ZG9fY29tcGFyZShjaGFyIGNvbnN0KiwgY2hhciBjb25zdCosIGNoYXIgY29uc3QqLCBjaGFyIGNvbnN0KikgY29uc3SKA0VzdGQ6Ol9fMjo6Y29sbGF0ZTxjaGFyPjo6ZG9fdHJhbnNmb3JtKGNoYXIgY29uc3QqLCBjaGFyIGNvbnN0KikgY29uc3SLA5sBc3RkOjpfXzI6OmJhc2ljX3N0cmluZzxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4sIHN0ZDo6X18yOjphbGxvY2F0b3I8Y2hhcj4+OjpiYXNpY19zdHJpbmdbYWJpOnYxNTAwN108Y2hhciBjb25zdCosIHZvaWQ+KGNoYXIgY29uc3QqLCBjaGFyIGNvbnN0KimMA0BzdGQ6Ol9fMjo6Y29sbGF0ZTxjaGFyPjo6ZG9faGFzaChjaGFyIGNvbnN0KiwgY2hhciBjb25zdCopIGNvbnN0jQNsc3RkOjpfXzI6OmNvbGxhdGU8d2NoYXJfdD46OmRvX2NvbXBhcmUod2NoYXJfdCBjb25zdCosIHdjaGFyX3QgY29uc3QqLCB3Y2hhcl90IGNvbnN0Kiwgd2NoYXJfdCBjb25zdCopIGNvbnN0jgNOc3RkOjpfXzI6OmNvbGxhdGU8d2NoYXJfdD46OmRvX3RyYW5zZm9ybSh3Y2hhcl90IGNvbnN0Kiwgd2NoYXJfdCBjb25zdCopIGNvbnN0jwPpAXN0ZDo6X18yOjplbmFibGVfaWY8X19pc19jcHAxN19mb3J3YXJkX2l0ZXJhdG9yPHdjaGFyX3QgY29uc3QqPjo6dmFsdWUsIHZvaWQ+Ojp0eXBlIHN0ZDo6X18yOjpiYXNpY19zdHJpbmc8d2NoYXJfdCwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPHdjaGFyX3Q+LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPHdjaGFyX3Q+Pjo6X19pbml0PHdjaGFyX3QgY29uc3QqPih3Y2hhcl90IGNvbnN0Kiwgd2NoYXJfdCBjb25zdCopkANJc3RkOjpfXzI6OmNvbGxhdGU8d2NoYXJfdD46OmRvX2hhc2god2NoYXJfdCBjb25zdCosIHdjaGFyX3QgY29uc3QqKSBjb25zdJEDlgJzdGQ6Ol9fMjo6bnVtX2dldDxjaGFyLCBzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+Pjo6ZG9fZ2V0KHN0ZDo6X18yOjppc3RyZWFtYnVmX2l0ZXJhdG9yPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPj4sIHN0ZDo6X18yOjppc3RyZWFtYnVmX2l0ZXJhdG9yPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPj4sIHN0ZDo6X18yOjppb3NfYmFzZSYsIHVuc2lnbmVkIGludCYsIGJvb2wmKSBjb25zdJIDcnN0ZDo6X18yOjpudW1wdW5jdDxjaGFyPiBjb25zdCYgc3RkOjpfXzI6OnVzZV9mYWNldFthYmk6djE1MDA3XTxzdGQ6Ol9fMjo6bnVtcHVuY3Q8Y2hhcj4+KHN0ZDo6X18yOjpsb2NhbGUgY29uc3QmKZMDnAVzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPiwgc3RkOjpfXzI6OmFsbG9jYXRvcjxjaGFyPj4gY29uc3QqIHN0ZDo6X18yOjpfX3NjYW5fa2V5d29yZDxzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+LCBzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPiwgc3RkOjpfXzI6OmFsbG9jYXRvcjxjaGFyPj4gY29uc3QqLCBzdGQ6Ol9fMjo6Y3R5cGU8Y2hhcj4+KHN0ZDo6X18yOjppc3RyZWFtYnVmX2l0ZXJhdG9yPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPj4mLCBzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+LCBzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPiwgc3RkOjpfXzI6OmFsbG9jYXRvcjxjaGFyPj4gY29uc3QqLCBzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPiwgc3RkOjpfXzI6OmFsbG9jYXRvcjxjaGFyPj4gY29uc3QqLCBzdGQ6Ol9fMjo6Y3R5cGU8Y2hhcj4gY29uc3QmLCB1bnNpZ25lZCBpbnQmLCBib29sKZQDOHN0ZDo6X18yOjpsb2NhbGU6OnVzZV9mYWNldChzdGQ6Ol9fMjo6bG9jYWxlOjppZCYpIGNvbnN0lQNXc3RkOjpfXzI6OnVuaXF1ZV9wdHI8dW5zaWduZWQgY2hhciwgdm9pZCAoKikodm9pZCopPjo6cmVzZXRbYWJpOnYxNTAwN10odW5zaWduZWQgY2hhcioplgNPc3RkOjpfXzI6OnVuaXF1ZV9wdHI8dW5zaWduZWQgY2hhciwgdm9pZCAoKikodm9pZCopPjo6fnVuaXF1ZV9wdHJbYWJpOnYxNTAwN10oKZcDlgJzdGQ6Ol9fMjo6bnVtX2dldDxjaGFyLCBzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+Pjo6ZG9fZ2V0KHN0ZDo6X18yOjppc3RyZWFtYnVmX2l0ZXJhdG9yPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPj4sIHN0ZDo6X18yOjppc3RyZWFtYnVmX2l0ZXJhdG9yPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPj4sIHN0ZDo6X18yOjppb3NfYmFzZSYsIHVuc2lnbmVkIGludCYsIGxvbmcmKSBjb25zdJgDOXN0ZDo6X18yOjpfX251bV9nZXRfYmFzZTo6X19nZXRfYmFzZShzdGQ6Ol9fMjo6aW9zX2Jhc2UmKZkDSHN0ZDo6X18yOjpfX251bV9nZXQ8Y2hhcj46Ol9fc3RhZ2UyX2ludF9wcmVwKHN0ZDo6X18yOjppb3NfYmFzZSYsIGNoYXImKZoD5AFzdGQ6Ol9fMjo6X19udW1fZ2V0PGNoYXI+OjpfX3N0YWdlMl9pbnRfbG9vcChjaGFyLCBpbnQsIGNoYXIqLCBjaGFyKiYsIHVuc2lnbmVkIGludCYsIGNoYXIsIHN0ZDo6X18yOjpiYXNpY19zdHJpbmc8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPGNoYXI+PiBjb25zdCYsIHVuc2lnbmVkIGludCosIHVuc2lnbmVkIGludComLCBjaGFyIGNvbnN0KimbA1xsb25nIHN0ZDo6X18yOjpfX251bV9nZXRfc2lnbmVkX2ludGVncmFsPGxvbmc+KGNoYXIgY29uc3QqLCBjaGFyIGNvbnN0KiwgdW5zaWduZWQgaW50JiwgaW50KZwDpAFzdGQ6Ol9fMjo6X19jaGVja19ncm91cGluZyhzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPiwgc3RkOjpfXzI6OmFsbG9jYXRvcjxjaGFyPj4gY29uc3QmLCB1bnNpZ25lZCBpbnQqLCB1bnNpZ25lZCBpbnQqLCB1bnNpZ25lZCBpbnQmKZ0DmwJzdGQ6Ol9fMjo6bnVtX2dldDxjaGFyLCBzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+Pjo6ZG9fZ2V0KHN0ZDo6X18yOjppc3RyZWFtYnVmX2l0ZXJhdG9yPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPj4sIHN0ZDo6X18yOjppc3RyZWFtYnVmX2l0ZXJhdG9yPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPj4sIHN0ZDo6X18yOjppb3NfYmFzZSYsIHVuc2lnbmVkIGludCYsIGxvbmcgbG9uZyYpIGNvbnN0ngNmbG9uZyBsb25nIHN0ZDo6X18yOjpfX251bV9nZXRfc2lnbmVkX2ludGVncmFsPGxvbmcgbG9uZz4oY2hhciBjb25zdCosIGNoYXIgY29uc3QqLCB1bnNpZ25lZCBpbnQmLCBpbnQpnwOgAnN0ZDo6X18yOjpudW1fZ2V0PGNoYXIsIHN0ZDo6X18yOjppc3RyZWFtYnVmX2l0ZXJhdG9yPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPj4+Ojpkb19nZXQoc3RkOjpfXzI6OmlzdHJlYW1idWZfaXRlcmF0b3I8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+Piwgc3RkOjpfXzI6OmlzdHJlYW1idWZfaXRlcmF0b3I8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+Piwgc3RkOjpfXzI6Omlvc19iYXNlJiwgdW5zaWduZWQgaW50JiwgdW5zaWduZWQgc2hvcnQmKSBjb25zdKADcnVuc2lnbmVkIHNob3J0IHN0ZDo6X18yOjpfX251bV9nZXRfdW5zaWduZWRfaW50ZWdyYWw8dW5zaWduZWQgc2hvcnQ+KGNoYXIgY29uc3QqLCBjaGFyIGNvbnN0KiwgdW5zaWduZWQgaW50JiwgaW50KaEDngJzdGQ6Ol9fMjo6bnVtX2dldDxjaGFyLCBzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+Pjo6ZG9fZ2V0KHN0ZDo6X18yOjppc3RyZWFtYnVmX2l0ZXJhdG9yPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPj4sIHN0ZDo6X18yOjppc3RyZWFtYnVmX2l0ZXJhdG9yPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPj4sIHN0ZDo6X18yOjppb3NfYmFzZSYsIHVuc2lnbmVkIGludCYsIHVuc2lnbmVkIGludCYpIGNvbnN0ogNudW5zaWduZWQgaW50IHN0ZDo6X18yOjpfX251bV9nZXRfdW5zaWduZWRfaW50ZWdyYWw8dW5zaWduZWQgaW50PihjaGFyIGNvbnN0KiwgY2hhciBjb25zdCosIHVuc2lnbmVkIGludCYsIGludCmjA6QCc3RkOjpfXzI6Om51bV9nZXQ8Y2hhciwgc3RkOjpfXzI6OmlzdHJlYW1idWZfaXRlcmF0b3I8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+Pj46OmRvX2dldChzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+LCBzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+LCBzdGQ6Ol9fMjo6aW9zX2Jhc2UmLCB1bnNpZ25lZCBpbnQmLCB1bnNpZ25lZCBsb25nIGxvbmcmKSBjb25zdKQDenVuc2lnbmVkIGxvbmcgbG9uZyBzdGQ6Ol9fMjo6X19udW1fZ2V0X3Vuc2lnbmVkX2ludGVncmFsPHVuc2lnbmVkIGxvbmcgbG9uZz4oY2hhciBjb25zdCosIGNoYXIgY29uc3QqLCB1bnNpZ25lZCBpbnQmLCBpbnQppQOXAnN0ZDo6X18yOjpudW1fZ2V0PGNoYXIsIHN0ZDo6X18yOjppc3RyZWFtYnVmX2l0ZXJhdG9yPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPj4+Ojpkb19nZXQoc3RkOjpfXzI6OmlzdHJlYW1idWZfaXRlcmF0b3I8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+Piwgc3RkOjpfXzI6OmlzdHJlYW1idWZfaXRlcmF0b3I8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+Piwgc3RkOjpfXzI6Omlvc19iYXNlJiwgdW5zaWduZWQgaW50JiwgZmxvYXQmKSBjb25zdKYDWHN0ZDo6X18yOjpfX251bV9nZXQ8Y2hhcj46Ol9fc3RhZ2UyX2Zsb2F0X3ByZXAoc3RkOjpfXzI6Omlvc19iYXNlJiwgY2hhciosIGNoYXImLCBjaGFyJimnA+8Bc3RkOjpfXzI6Ol9fbnVtX2dldDxjaGFyPjo6X19zdGFnZTJfZmxvYXRfbG9vcChjaGFyLCBib29sJiwgY2hhciYsIGNoYXIqLCBjaGFyKiYsIGNoYXIsIGNoYXIsIHN0ZDo6X18yOjpiYXNpY19zdHJpbmc8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPGNoYXI+PiBjb25zdCYsIHVuc2lnbmVkIGludCosIHVuc2lnbmVkIGludComLCB1bnNpZ25lZCBpbnQmLCBjaGFyKimoA09mbG9hdCBzdGQ6Ol9fMjo6X19udW1fZ2V0X2Zsb2F0PGZsb2F0PihjaGFyIGNvbnN0KiwgY2hhciBjb25zdCosIHVuc2lnbmVkIGludCYpqQOYAnN0ZDo6X18yOjpudW1fZ2V0PGNoYXIsIHN0ZDo6X18yOjppc3RyZWFtYnVmX2l0ZXJhdG9yPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPj4+Ojpkb19nZXQoc3RkOjpfXzI6OmlzdHJlYW1idWZfaXRlcmF0b3I8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+Piwgc3RkOjpfXzI6OmlzdHJlYW1idWZfaXRlcmF0b3I8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+Piwgc3RkOjpfXzI6Omlvc19iYXNlJiwgdW5zaWduZWQgaW50JiwgZG91YmxlJikgY29uc3SqA1Fkb3VibGUgc3RkOjpfXzI6Ol9fbnVtX2dldF9mbG9hdDxkb3VibGU+KGNoYXIgY29uc3QqLCBjaGFyIGNvbnN0KiwgdW5zaWduZWQgaW50JimrA50Cc3RkOjpfXzI6Om51bV9nZXQ8Y2hhciwgc3RkOjpfXzI6OmlzdHJlYW1idWZfaXRlcmF0b3I8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+Pj46OmRvX2dldChzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+LCBzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+LCBzdGQ6Ol9fMjo6aW9zX2Jhc2UmLCB1bnNpZ25lZCBpbnQmLCBsb25nIGRvdWJsZSYpIGNvbnN0rANbbG9uZyBkb3VibGUgc3RkOjpfXzI6Ol9fbnVtX2dldF9mbG9hdDxsb25nIGRvdWJsZT4oY2hhciBjb25zdCosIGNoYXIgY29uc3QqLCB1bnNpZ25lZCBpbnQmKa0DlwJzdGQ6Ol9fMjo6bnVtX2dldDxjaGFyLCBzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+Pjo6ZG9fZ2V0KHN0ZDo6X18yOjppc3RyZWFtYnVmX2l0ZXJhdG9yPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPj4sIHN0ZDo6X18yOjppc3RyZWFtYnVmX2l0ZXJhdG9yPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPj4sIHN0ZDo6X18yOjppb3NfYmFzZSYsIHVuc2lnbmVkIGludCYsIHZvaWQqJikgY29uc3SuAxJzdGQ6Ol9fMjo6X19jbG9jKCmvA0xzdGQ6Ol9fMjo6X19saWJjcHBfc3NjYW5mX2woY2hhciBjb25zdCosIF9fbG9jYWxlX3N0cnVjdCosIGNoYXIgY29uc3QqLCAuLi4psANgY2hhciBjb25zdCogc3RkOjpfXzI6OmZpbmRbYWJpOnYxNTAwN108Y2hhciBjb25zdCosIGNoYXI+KGNoYXIgY29uc3QqLCBjaGFyIGNvbnN0KiwgY2hhciBjb25zdCYpsQNVc3RkOjpfXzI6Ol9fbGliY3BwX2xvY2FsZV9ndWFyZDo6X19saWJjcHBfbG9jYWxlX2d1YXJkW2FiaTp2MTUwMDddKF9fbG9jYWxlX3N0cnVjdComKbIDqwJzdGQ6Ol9fMjo6bnVtX2dldDx3Y2hhcl90LCBzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjx3Y2hhcl90LCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8d2NoYXJfdD4+Pjo6ZG9fZ2V0KHN0ZDo6X18yOjppc3RyZWFtYnVmX2l0ZXJhdG9yPHdjaGFyX3QsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czx3Y2hhcl90Pj4sIHN0ZDo6X18yOjppc3RyZWFtYnVmX2l0ZXJhdG9yPHdjaGFyX3QsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czx3Y2hhcl90Pj4sIHN0ZDo6X18yOjppb3NfYmFzZSYsIHVuc2lnbmVkIGludCYsIGJvb2wmKSBjb25zdLMDeHN0ZDo6X18yOjpudW1wdW5jdDx3Y2hhcl90PiBjb25zdCYgc3RkOjpfXzI6OnVzZV9mYWNldFthYmk6djE1MDA3XTxzdGQ6Ol9fMjo6bnVtcHVuY3Q8d2NoYXJfdD4+KHN0ZDo6X18yOjpsb2NhbGUgY29uc3QmKbQD2AVzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPHdjaGFyX3QsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czx3Y2hhcl90Piwgc3RkOjpfXzI6OmFsbG9jYXRvcjx3Y2hhcl90Pj4gY29uc3QqIHN0ZDo6X18yOjpfX3NjYW5fa2V5d29yZDxzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjx3Y2hhcl90LCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8d2NoYXJfdD4+LCBzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPHdjaGFyX3QsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czx3Y2hhcl90Piwgc3RkOjpfXzI6OmFsbG9jYXRvcjx3Y2hhcl90Pj4gY29uc3QqLCBzdGQ6Ol9fMjo6Y3R5cGU8d2NoYXJfdD4+KHN0ZDo6X18yOjppc3RyZWFtYnVmX2l0ZXJhdG9yPHdjaGFyX3QsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czx3Y2hhcl90Pj4mLCBzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjx3Y2hhcl90LCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8d2NoYXJfdD4+LCBzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPHdjaGFyX3QsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czx3Y2hhcl90Piwgc3RkOjpfXzI6OmFsbG9jYXRvcjx3Y2hhcl90Pj4gY29uc3QqLCBzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPHdjaGFyX3QsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czx3Y2hhcl90Piwgc3RkOjpfXzI6OmFsbG9jYXRvcjx3Y2hhcl90Pj4gY29uc3QqLCBzdGQ6Ol9fMjo6Y3R5cGU8d2NoYXJfdD4gY29uc3QmLCB1bnNpZ25lZCBpbnQmLCBib29sKbUDqwJzdGQ6Ol9fMjo6bnVtX2dldDx3Y2hhcl90LCBzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjx3Y2hhcl90LCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8d2NoYXJfdD4+Pjo6ZG9fZ2V0KHN0ZDo6X18yOjppc3RyZWFtYnVmX2l0ZXJhdG9yPHdjaGFyX3QsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czx3Y2hhcl90Pj4sIHN0ZDo6X18yOjppc3RyZWFtYnVmX2l0ZXJhdG9yPHdjaGFyX3QsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czx3Y2hhcl90Pj4sIHN0ZDo6X18yOjppb3NfYmFzZSYsIHVuc2lnbmVkIGludCYsIGxvbmcmKSBjb25zdLYDTXN0ZDo6X18yOjpfX251bV9nZXQ8d2NoYXJfdD46Ol9fZG9fd2lkZW4oc3RkOjpfXzI6Omlvc19iYXNlJiwgd2NoYXJfdCopIGNvbnN0twNOc3RkOjpfXzI6Ol9fbnVtX2dldDx3Y2hhcl90Pjo6X19zdGFnZTJfaW50X3ByZXAoc3RkOjpfXzI6Omlvc19iYXNlJiwgd2NoYXJfdCYpuAPwAXN0ZDo6X18yOjpfX251bV9nZXQ8d2NoYXJfdD46Ol9fc3RhZ2UyX2ludF9sb29wKHdjaGFyX3QsIGludCwgY2hhciosIGNoYXIqJiwgdW5zaWduZWQgaW50Jiwgd2NoYXJfdCwgc3RkOjpfXzI6OmJhc2ljX3N0cmluZzxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4sIHN0ZDo6X18yOjphbGxvY2F0b3I8Y2hhcj4+IGNvbnN0JiwgdW5zaWduZWQgaW50KiwgdW5zaWduZWQgaW50KiYsIHdjaGFyX3QgY29uc3QqKbkDsAJzdGQ6Ol9fMjo6bnVtX2dldDx3Y2hhcl90LCBzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjx3Y2hhcl90LCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8d2NoYXJfdD4+Pjo6ZG9fZ2V0KHN0ZDo6X18yOjppc3RyZWFtYnVmX2l0ZXJhdG9yPHdjaGFyX3QsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czx3Y2hhcl90Pj4sIHN0ZDo6X18yOjppc3RyZWFtYnVmX2l0ZXJhdG9yPHdjaGFyX3QsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czx3Y2hhcl90Pj4sIHN0ZDo6X18yOjppb3NfYmFzZSYsIHVuc2lnbmVkIGludCYsIGxvbmcgbG9uZyYpIGNvbnN0ugO1AnN0ZDo6X18yOjpudW1fZ2V0PHdjaGFyX3QsIHN0ZDo6X18yOjppc3RyZWFtYnVmX2l0ZXJhdG9yPHdjaGFyX3QsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czx3Y2hhcl90Pj4+Ojpkb19nZXQoc3RkOjpfXzI6OmlzdHJlYW1idWZfaXRlcmF0b3I8d2NoYXJfdCwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPHdjaGFyX3Q+Piwgc3RkOjpfXzI6OmlzdHJlYW1idWZfaXRlcmF0b3I8d2NoYXJfdCwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPHdjaGFyX3Q+Piwgc3RkOjpfXzI6Omlvc19iYXNlJiwgdW5zaWduZWQgaW50JiwgdW5zaWduZWQgc2hvcnQmKSBjb25zdLsDswJzdGQ6Ol9fMjo6bnVtX2dldDx3Y2hhcl90LCBzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjx3Y2hhcl90LCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8d2NoYXJfdD4+Pjo6ZG9fZ2V0KHN0ZDo6X18yOjppc3RyZWFtYnVmX2l0ZXJhdG9yPHdjaGFyX3QsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czx3Y2hhcl90Pj4sIHN0ZDo6X18yOjppc3RyZWFtYnVmX2l0ZXJhdG9yPHdjaGFyX3QsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czx3Y2hhcl90Pj4sIHN0ZDo6X18yOjppb3NfYmFzZSYsIHVuc2lnbmVkIGludCYsIHVuc2lnbmVkIGludCYpIGNvbnN0vAO5AnN0ZDo6X18yOjpudW1fZ2V0PHdjaGFyX3QsIHN0ZDo6X18yOjppc3RyZWFtYnVmX2l0ZXJhdG9yPHdjaGFyX3QsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czx3Y2hhcl90Pj4+Ojpkb19nZXQoc3RkOjpfXzI6OmlzdHJlYW1idWZfaXRlcmF0b3I8d2NoYXJfdCwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPHdjaGFyX3Q+Piwgc3RkOjpfXzI6OmlzdHJlYW1idWZfaXRlcmF0b3I8d2NoYXJfdCwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPHdjaGFyX3Q+Piwgc3RkOjpfXzI6Omlvc19iYXNlJiwgdW5zaWduZWQgaW50JiwgdW5zaWduZWQgbG9uZyBsb25nJikgY29uc3S9A6wCc3RkOjpfXzI6Om51bV9nZXQ8d2NoYXJfdCwgc3RkOjpfXzI6OmlzdHJlYW1idWZfaXRlcmF0b3I8d2NoYXJfdCwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPHdjaGFyX3Q+Pj46OmRvX2dldChzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjx3Y2hhcl90LCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8d2NoYXJfdD4+LCBzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjx3Y2hhcl90LCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8d2NoYXJfdD4+LCBzdGQ6Ol9fMjo6aW9zX2Jhc2UmLCB1bnNpZ25lZCBpbnQmLCBmbG9hdCYpIGNvbnN0vgNkc3RkOjpfXzI6Ol9fbnVtX2dldDx3Y2hhcl90Pjo6X19zdGFnZTJfZmxvYXRfcHJlcChzdGQ6Ol9fMjo6aW9zX2Jhc2UmLCB3Y2hhcl90Kiwgd2NoYXJfdCYsIHdjaGFyX3QmKb8D/gFzdGQ6Ol9fMjo6X19udW1fZ2V0PHdjaGFyX3Q+OjpfX3N0YWdlMl9mbG9hdF9sb29wKHdjaGFyX3QsIGJvb2wmLCBjaGFyJiwgY2hhciosIGNoYXIqJiwgd2NoYXJfdCwgd2NoYXJfdCwgc3RkOjpfXzI6OmJhc2ljX3N0cmluZzxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4sIHN0ZDo6X18yOjphbGxvY2F0b3I8Y2hhcj4+IGNvbnN0JiwgdW5zaWduZWQgaW50KiwgdW5zaWduZWQgaW50KiYsIHVuc2lnbmVkIGludCYsIHdjaGFyX3QqKcADrQJzdGQ6Ol9fMjo6bnVtX2dldDx3Y2hhcl90LCBzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjx3Y2hhcl90LCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8d2NoYXJfdD4+Pjo6ZG9fZ2V0KHN0ZDo6X18yOjppc3RyZWFtYnVmX2l0ZXJhdG9yPHdjaGFyX3QsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czx3Y2hhcl90Pj4sIHN0ZDo6X18yOjppc3RyZWFtYnVmX2l0ZXJhdG9yPHdjaGFyX3QsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czx3Y2hhcl90Pj4sIHN0ZDo6X18yOjppb3NfYmFzZSYsIHVuc2lnbmVkIGludCYsIGRvdWJsZSYpIGNvbnN0wQOyAnN0ZDo6X18yOjpudW1fZ2V0PHdjaGFyX3QsIHN0ZDo6X18yOjppc3RyZWFtYnVmX2l0ZXJhdG9yPHdjaGFyX3QsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czx3Y2hhcl90Pj4+Ojpkb19nZXQoc3RkOjpfXzI6OmlzdHJlYW1idWZfaXRlcmF0b3I8d2NoYXJfdCwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPHdjaGFyX3Q+Piwgc3RkOjpfXzI6OmlzdHJlYW1idWZfaXRlcmF0b3I8d2NoYXJfdCwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPHdjaGFyX3Q+Piwgc3RkOjpfXzI6Omlvc19iYXNlJiwgdW5zaWduZWQgaW50JiwgbG9uZyBkb3VibGUmKSBjb25zdMIDrAJzdGQ6Ol9fMjo6bnVtX2dldDx3Y2hhcl90LCBzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjx3Y2hhcl90LCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8d2NoYXJfdD4+Pjo6ZG9fZ2V0KHN0ZDo6X18yOjppc3RyZWFtYnVmX2l0ZXJhdG9yPHdjaGFyX3QsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czx3Y2hhcl90Pj4sIHN0ZDo6X18yOjppc3RyZWFtYnVmX2l0ZXJhdG9yPHdjaGFyX3QsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czx3Y2hhcl90Pj4sIHN0ZDo6X18yOjppb3NfYmFzZSYsIHVuc2lnbmVkIGludCYsIHZvaWQqJikgY29uc3TDA3J3Y2hhcl90IGNvbnN0KiBzdGQ6Ol9fMjo6ZmluZFthYmk6djE1MDA3XTx3Y2hhcl90IGNvbnN0Kiwgd2NoYXJfdD4od2NoYXJfdCBjb25zdCosIHdjaGFyX3QgY29uc3QqLCB3Y2hhcl90IGNvbnN0JinEA8oBc3RkOjpfXzI6Om51bV9wdXQ8Y2hhciwgc3RkOjpfXzI6Om9zdHJlYW1idWZfaXRlcmF0b3I8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+Pj46OmRvX3B1dChzdGQ6Ol9fMjo6b3N0cmVhbWJ1Zl9pdGVyYXRvcjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+LCBzdGQ6Ol9fMjo6aW9zX2Jhc2UmLCBjaGFyLCBib29sKSBjb25zdMUDaXN0ZDo6X18yOjpiYXNpY19zdHJpbmc8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPGNoYXI+Pjo6YmVnaW5bYWJpOnYxNTAwN10oKcYDZ3N0ZDo6X18yOjpiYXNpY19zdHJpbmc8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPGNoYXI+Pjo6ZW5kW2FiaTp2MTUwMDddKCnHA8oBc3RkOjpfXzI6Om51bV9wdXQ8Y2hhciwgc3RkOjpfXzI6Om9zdHJlYW1idWZfaXRlcmF0b3I8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+Pj46OmRvX3B1dChzdGQ6Ol9fMjo6b3N0cmVhbWJ1Zl9pdGVyYXRvcjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+LCBzdGQ6Ol9fMjo6aW9zX2Jhc2UmLCBjaGFyLCBsb25nKSBjb25zdMgDTnN0ZDo6X18yOjpfX251bV9wdXRfYmFzZTo6X19mb3JtYXRfaW50KGNoYXIqLCBjaGFyIGNvbnN0KiwgYm9vbCwgdW5zaWduZWQgaW50KckDV3N0ZDo6X18yOjpfX2xpYmNwcF9zbnByaW50Zl9sKGNoYXIqLCB1bnNpZ25lZCBsb25nLCBfX2xvY2FsZV9zdHJ1Y3QqLCBjaGFyIGNvbnN0KiwgLi4uKcoDVXN0ZDo6X18yOjpfX251bV9wdXRfYmFzZTo6X19pZGVudGlmeV9wYWRkaW5nKGNoYXIqLCBjaGFyKiwgc3RkOjpfXzI6Omlvc19iYXNlIGNvbnN0JinLA3VzdGQ6Ol9fMjo6X19udW1fcHV0PGNoYXI+OjpfX3dpZGVuX2FuZF9ncm91cF9pbnQoY2hhciosIGNoYXIqLCBjaGFyKiwgY2hhciosIGNoYXIqJiwgY2hhciomLCBzdGQ6Ol9fMjo6bG9jYWxlIGNvbnN0JinMA4ICc3RkOjpfXzI6Om9zdHJlYW1idWZfaXRlcmF0b3I8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+PiBzdGQ6Ol9fMjo6X19wYWRfYW5kX291dHB1dDxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+KHN0ZDo6X18yOjpvc3RyZWFtYnVmX2l0ZXJhdG9yPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPj4sIGNoYXIgY29uc3QqLCBjaGFyIGNvbnN0KiwgY2hhciBjb25zdCosIHN0ZDo6X18yOjppb3NfYmFzZSYsIGNoYXIpzQPPAXN0ZDo6X18yOjpudW1fcHV0PGNoYXIsIHN0ZDo6X18yOjpvc3RyZWFtYnVmX2l0ZXJhdG9yPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPj4+Ojpkb19wdXQoc3RkOjpfXzI6Om9zdHJlYW1idWZfaXRlcmF0b3I8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+Piwgc3RkOjpfXzI6Omlvc19iYXNlJiwgY2hhciwgbG9uZyBsb25nKSBjb25zdM4D0wFzdGQ6Ol9fMjo6bnVtX3B1dDxjaGFyLCBzdGQ6Ol9fMjo6b3N0cmVhbWJ1Zl9pdGVyYXRvcjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+Pjo6ZG9fcHV0KHN0ZDo6X18yOjpvc3RyZWFtYnVmX2l0ZXJhdG9yPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPj4sIHN0ZDo6X18yOjppb3NfYmFzZSYsIGNoYXIsIHVuc2lnbmVkIGxvbmcpIGNvbnN0zwPYAXN0ZDo6X18yOjpudW1fcHV0PGNoYXIsIHN0ZDo6X18yOjpvc3RyZWFtYnVmX2l0ZXJhdG9yPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPj4+Ojpkb19wdXQoc3RkOjpfXzI6Om9zdHJlYW1idWZfaXRlcmF0b3I8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+Piwgc3RkOjpfXzI6Omlvc19iYXNlJiwgY2hhciwgdW5zaWduZWQgbG9uZyBsb25nKSBjb25zdNADzAFzdGQ6Ol9fMjo6bnVtX3B1dDxjaGFyLCBzdGQ6Ol9fMjo6b3N0cmVhbWJ1Zl9pdGVyYXRvcjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+Pjo6ZG9fcHV0KHN0ZDo6X18yOjpvc3RyZWFtYnVmX2l0ZXJhdG9yPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPj4sIHN0ZDo6X18yOjppb3NfYmFzZSYsIGNoYXIsIGRvdWJsZSkgY29uc3TRA78Cc3RkOjpfXzI6Om9zdHJlYW1idWZfaXRlcmF0b3I8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+PiBzdGQ6Ol9fMjo6bnVtX3B1dDxjaGFyLCBzdGQ6Ol9fMjo6b3N0cmVhbWJ1Zl9pdGVyYXRvcjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+Pjo6X19kb19wdXRfZmxvYXRpbmdfcG9pbnRbYWJpOnYxNTAwN108ZG91YmxlPihzdGQ6Ol9fMjo6b3N0cmVhbWJ1Zl9pdGVyYXRvcjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+LCBzdGQ6Ol9fMjo6aW9zX2Jhc2UmLCBjaGFyLCBkb3VibGUsIGNoYXIgY29uc3QqKSBjb25zdNIDSnN0ZDo6X18yOjpfX251bV9wdXRfYmFzZTo6X19mb3JtYXRfZmxvYXQoY2hhciosIGNoYXIgY29uc3QqLCB1bnNpZ25lZCBpbnQp0wNJc3RkOjpfXzI6Ol9fbGliY3BwX2FzcHJpbnRmX2woY2hhcioqLCBfX2xvY2FsZV9zdHJ1Y3QqLCBjaGFyIGNvbnN0KiwgLi4uKdQDd3N0ZDo6X18yOjpfX251bV9wdXQ8Y2hhcj46Ol9fd2lkZW5fYW5kX2dyb3VwX2Zsb2F0KGNoYXIqLCBjaGFyKiwgY2hhciosIGNoYXIqLCBjaGFyKiYsIGNoYXIqJiwgc3RkOjpfXzI6OmxvY2FsZSBjb25zdCYp1QPRAXN0ZDo6X18yOjpudW1fcHV0PGNoYXIsIHN0ZDo6X18yOjpvc3RyZWFtYnVmX2l0ZXJhdG9yPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPj4+Ojpkb19wdXQoc3RkOjpfXzI6Om9zdHJlYW1idWZfaXRlcmF0b3I8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+Piwgc3RkOjpfXzI6Omlvc19iYXNlJiwgY2hhciwgbG9uZyBkb3VibGUpIGNvbnN01gPJAnN0ZDo6X18yOjpvc3RyZWFtYnVmX2l0ZXJhdG9yPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPj4gc3RkOjpfXzI6Om51bV9wdXQ8Y2hhciwgc3RkOjpfXzI6Om9zdHJlYW1idWZfaXRlcmF0b3I8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+Pj46Ol9fZG9fcHV0X2Zsb2F0aW5nX3BvaW50W2FiaTp2MTUwMDddPGxvbmcgZG91YmxlPihzdGQ6Ol9fMjo6b3N0cmVhbWJ1Zl9pdGVyYXRvcjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+LCBzdGQ6Ol9fMjo6aW9zX2Jhc2UmLCBjaGFyLCBsb25nIGRvdWJsZSwgY2hhciBjb25zdCopIGNvbnN01wPRAXN0ZDo6X18yOjpudW1fcHV0PGNoYXIsIHN0ZDo6X18yOjpvc3RyZWFtYnVmX2l0ZXJhdG9yPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPj4+Ojpkb19wdXQoc3RkOjpfXzI6Om9zdHJlYW1idWZfaXRlcmF0b3I8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+Piwgc3RkOjpfXzI6Omlvc19iYXNlJiwgY2hhciwgdm9pZCBjb25zdCopIGNvbnN02AODAXN0ZDo6X18yOjpiYXNpY19zdHJpbmc8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPGNoYXI+Pjo6YmFzaWNfc3RyaW5nW2FiaTp2MTUwMDddKHVuc2lnbmVkIGxvbmcsIGNoYXIp2QPcAXN0ZDo6X18yOjpudW1fcHV0PHdjaGFyX3QsIHN0ZDo6X18yOjpvc3RyZWFtYnVmX2l0ZXJhdG9yPHdjaGFyX3QsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czx3Y2hhcl90Pj4+Ojpkb19wdXQoc3RkOjpfXzI6Om9zdHJlYW1idWZfaXRlcmF0b3I8d2NoYXJfdCwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPHdjaGFyX3Q+Piwgc3RkOjpfXzI6Omlvc19iYXNlJiwgd2NoYXJfdCwgYm9vbCkgY29uc3TaA3BzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPHdjaGFyX3QsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czx3Y2hhcl90Piwgc3RkOjpfXzI6OmFsbG9jYXRvcjx3Y2hhcl90Pj46OmVuZFthYmk6djE1MDA3XSgp2wPcAXN0ZDo6X18yOjpudW1fcHV0PHdjaGFyX3QsIHN0ZDo6X18yOjpvc3RyZWFtYnVmX2l0ZXJhdG9yPHdjaGFyX3QsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czx3Y2hhcl90Pj4+Ojpkb19wdXQoc3RkOjpfXzI6Om9zdHJlYW1idWZfaXRlcmF0b3I8d2NoYXJfdCwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPHdjaGFyX3Q+Piwgc3RkOjpfXzI6Omlvc19iYXNlJiwgd2NoYXJfdCwgbG9uZykgY29uc3TcA4EBc3RkOjpfXzI6Ol9fbnVtX3B1dDx3Y2hhcl90Pjo6X193aWRlbl9hbmRfZ3JvdXBfaW50KGNoYXIqLCBjaGFyKiwgY2hhciosIHdjaGFyX3QqLCB3Y2hhcl90KiYsIHdjaGFyX3QqJiwgc3RkOjpfXzI6OmxvY2FsZSBjb25zdCYp3QOgAnN0ZDo6X18yOjpvc3RyZWFtYnVmX2l0ZXJhdG9yPHdjaGFyX3QsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czx3Y2hhcl90Pj4gc3RkOjpfXzI6Ol9fcGFkX2FuZF9vdXRwdXQ8d2NoYXJfdCwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPHdjaGFyX3Q+PihzdGQ6Ol9fMjo6b3N0cmVhbWJ1Zl9pdGVyYXRvcjx3Y2hhcl90LCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8d2NoYXJfdD4+LCB3Y2hhcl90IGNvbnN0Kiwgd2NoYXJfdCBjb25zdCosIHdjaGFyX3QgY29uc3QqLCBzdGQ6Ol9fMjo6aW9zX2Jhc2UmLCB3Y2hhcl90Kd4D4QFzdGQ6Ol9fMjo6bnVtX3B1dDx3Y2hhcl90LCBzdGQ6Ol9fMjo6b3N0cmVhbWJ1Zl9pdGVyYXRvcjx3Y2hhcl90LCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8d2NoYXJfdD4+Pjo6ZG9fcHV0KHN0ZDo6X18yOjpvc3RyZWFtYnVmX2l0ZXJhdG9yPHdjaGFyX3QsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czx3Y2hhcl90Pj4sIHN0ZDo6X18yOjppb3NfYmFzZSYsIHdjaGFyX3QsIGxvbmcgbG9uZykgY29uc3TfA+UBc3RkOjpfXzI6Om51bV9wdXQ8d2NoYXJfdCwgc3RkOjpfXzI6Om9zdHJlYW1idWZfaXRlcmF0b3I8d2NoYXJfdCwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPHdjaGFyX3Q+Pj46OmRvX3B1dChzdGQ6Ol9fMjo6b3N0cmVhbWJ1Zl9pdGVyYXRvcjx3Y2hhcl90LCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8d2NoYXJfdD4+LCBzdGQ6Ol9fMjo6aW9zX2Jhc2UmLCB3Y2hhcl90LCB1bnNpZ25lZCBsb25nKSBjb25zdOAD6gFzdGQ6Ol9fMjo6bnVtX3B1dDx3Y2hhcl90LCBzdGQ6Ol9fMjo6b3N0cmVhbWJ1Zl9pdGVyYXRvcjx3Y2hhcl90LCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8d2NoYXJfdD4+Pjo6ZG9fcHV0KHN0ZDo6X18yOjpvc3RyZWFtYnVmX2l0ZXJhdG9yPHdjaGFyX3QsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czx3Y2hhcl90Pj4sIHN0ZDo6X18yOjppb3NfYmFzZSYsIHdjaGFyX3QsIHVuc2lnbmVkIGxvbmcgbG9uZykgY29uc3ThA94Bc3RkOjpfXzI6Om51bV9wdXQ8d2NoYXJfdCwgc3RkOjpfXzI6Om9zdHJlYW1idWZfaXRlcmF0b3I8d2NoYXJfdCwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPHdjaGFyX3Q+Pj46OmRvX3B1dChzdGQ6Ol9fMjo6b3N0cmVhbWJ1Zl9pdGVyYXRvcjx3Y2hhcl90LCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8d2NoYXJfdD4+LCBzdGQ6Ol9fMjo6aW9zX2Jhc2UmLCB3Y2hhcl90LCBkb3VibGUpIGNvbnN04gPXAnN0ZDo6X18yOjpvc3RyZWFtYnVmX2l0ZXJhdG9yPHdjaGFyX3QsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czx3Y2hhcl90Pj4gc3RkOjpfXzI6Om51bV9wdXQ8d2NoYXJfdCwgc3RkOjpfXzI6Om9zdHJlYW1idWZfaXRlcmF0b3I8d2NoYXJfdCwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPHdjaGFyX3Q+Pj46Ol9fZG9fcHV0X2Zsb2F0aW5nX3BvaW50W2FiaTp2MTUwMDddPGRvdWJsZT4oc3RkOjpfXzI6Om9zdHJlYW1idWZfaXRlcmF0b3I8d2NoYXJfdCwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPHdjaGFyX3Q+Piwgc3RkOjpfXzI6Omlvc19iYXNlJiwgd2NoYXJfdCwgZG91YmxlLCBjaGFyIGNvbnN0KikgY29uc3TjA4MBc3RkOjpfXzI6Ol9fbnVtX3B1dDx3Y2hhcl90Pjo6X193aWRlbl9hbmRfZ3JvdXBfZmxvYXQoY2hhciosIGNoYXIqLCBjaGFyKiwgd2NoYXJfdCosIHdjaGFyX3QqJiwgd2NoYXJfdComLCBzdGQ6Ol9fMjo6bG9jYWxlIGNvbnN0JinkA+MBc3RkOjpfXzI6Om51bV9wdXQ8d2NoYXJfdCwgc3RkOjpfXzI6Om9zdHJlYW1idWZfaXRlcmF0b3I8d2NoYXJfdCwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPHdjaGFyX3Q+Pj46OmRvX3B1dChzdGQ6Ol9fMjo6b3N0cmVhbWJ1Zl9pdGVyYXRvcjx3Y2hhcl90LCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8d2NoYXJfdD4+LCBzdGQ6Ol9fMjo6aW9zX2Jhc2UmLCB3Y2hhcl90LCBsb25nIGRvdWJsZSkgY29uc3TlA+ECc3RkOjpfXzI6Om9zdHJlYW1idWZfaXRlcmF0b3I8d2NoYXJfdCwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPHdjaGFyX3Q+PiBzdGQ6Ol9fMjo6bnVtX3B1dDx3Y2hhcl90LCBzdGQ6Ol9fMjo6b3N0cmVhbWJ1Zl9pdGVyYXRvcjx3Y2hhcl90LCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8d2NoYXJfdD4+Pjo6X19kb19wdXRfZmxvYXRpbmdfcG9pbnRbYWJpOnYxNTAwN108bG9uZyBkb3VibGU+KHN0ZDo6X18yOjpvc3RyZWFtYnVmX2l0ZXJhdG9yPHdjaGFyX3QsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czx3Y2hhcl90Pj4sIHN0ZDo6X18yOjppb3NfYmFzZSYsIHdjaGFyX3QsIGxvbmcgZG91YmxlLCBjaGFyIGNvbnN0KikgY29uc3TmA+MBc3RkOjpfXzI6Om51bV9wdXQ8d2NoYXJfdCwgc3RkOjpfXzI6Om9zdHJlYW1idWZfaXRlcmF0b3I8d2NoYXJfdCwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPHdjaGFyX3Q+Pj46OmRvX3B1dChzdGQ6Ol9fMjo6b3N0cmVhbWJ1Zl9pdGVyYXRvcjx3Y2hhcl90LCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8d2NoYXJfdD4+LCBzdGQ6Ol9fMjo6aW9zX2Jhc2UmLCB3Y2hhcl90LCB2b2lkIGNvbnN0KikgY29uc3TnA48Bc3RkOjpfXzI6OmJhc2ljX3N0cmluZzx3Y2hhcl90LCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8d2NoYXJfdD4sIHN0ZDo6X18yOjphbGxvY2F0b3I8d2NoYXJfdD4+OjpiYXNpY19zdHJpbmdbYWJpOnYxNTAwN10odW5zaWduZWQgbG9uZywgd2NoYXJfdCnoAzd2b2lkIHN0ZDo6X18yOjpyZXZlcnNlW2FiaTp2MTUwMDddPGNoYXIqPihjaGFyKiwgY2hhciop6QNAdm9pZCBzdGQ6Ol9fMjo6cmV2ZXJzZVthYmk6djE1MDA3XTx3Y2hhcl90Kj4od2NoYXJfdCosIHdjaGFyX3QqKeoDrAJzdGQ6Ol9fMjo6dGltZV9nZXQ8Y2hhciwgc3RkOjpfXzI6OmlzdHJlYW1idWZfaXRlcmF0b3I8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+Pj46OmdldChzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+LCBzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+LCBzdGQ6Ol9fMjo6aW9zX2Jhc2UmLCB1bnNpZ25lZCBpbnQmLCB0bSosIGNoYXIgY29uc3QqLCBjaGFyIGNvbnN0KikgY29uc3TrA3FzdGQ6Ol9fMjo6dGltZV9nZXQ8Y2hhciwgc3RkOjpfXzI6OmlzdHJlYW1idWZfaXRlcmF0b3I8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+Pj46OmRvX2RhdGVfb3JkZXIoKSBjb25zdOwDmgJzdGQ6Ol9fMjo6dGltZV9nZXQ8Y2hhciwgc3RkOjpfXzI6OmlzdHJlYW1idWZfaXRlcmF0b3I8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+Pj46OmRvX2dldF90aW1lKHN0ZDo6X18yOjppc3RyZWFtYnVmX2l0ZXJhdG9yPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPj4sIHN0ZDo6X18yOjppc3RyZWFtYnVmX2l0ZXJhdG9yPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPj4sIHN0ZDo6X18yOjppb3NfYmFzZSYsIHVuc2lnbmVkIGludCYsIHRtKikgY29uc3TtA5oCc3RkOjpfXzI6OnRpbWVfZ2V0PGNoYXIsIHN0ZDo6X18yOjppc3RyZWFtYnVmX2l0ZXJhdG9yPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPj4+Ojpkb19nZXRfZGF0ZShzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+LCBzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+LCBzdGQ6Ol9fMjo6aW9zX2Jhc2UmLCB1bnNpZ25lZCBpbnQmLCB0bSopIGNvbnN07gOdAnN0ZDo6X18yOjp0aW1lX2dldDxjaGFyLCBzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+Pjo6ZG9fZ2V0X3dlZWtkYXkoc3RkOjpfXzI6OmlzdHJlYW1idWZfaXRlcmF0b3I8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+Piwgc3RkOjpfXzI6OmlzdHJlYW1idWZfaXRlcmF0b3I8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+Piwgc3RkOjpfXzI6Omlvc19iYXNlJiwgdW5zaWduZWQgaW50JiwgdG0qKSBjb25zdO8DqwJzdGQ6Ol9fMjo6dGltZV9nZXQ8Y2hhciwgc3RkOjpfXzI6OmlzdHJlYW1idWZfaXRlcmF0b3I8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+Pj46Ol9fZ2V0X3dlZWtkYXluYW1lKGludCYsIHN0ZDo6X18yOjppc3RyZWFtYnVmX2l0ZXJhdG9yPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPj4mLCBzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+LCB1bnNpZ25lZCBpbnQmLCBzdGQ6Ol9fMjo6Y3R5cGU8Y2hhcj4gY29uc3QmKSBjb25zdPADnwJzdGQ6Ol9fMjo6dGltZV9nZXQ8Y2hhciwgc3RkOjpfXzI6OmlzdHJlYW1idWZfaXRlcmF0b3I8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+Pj46OmRvX2dldF9tb250aG5hbWUoc3RkOjpfXzI6OmlzdHJlYW1idWZfaXRlcmF0b3I8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+Piwgc3RkOjpfXzI6OmlzdHJlYW1idWZfaXRlcmF0b3I8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+Piwgc3RkOjpfXzI6Omlvc19iYXNlJiwgdW5zaWduZWQgaW50JiwgdG0qKSBjb25zdPEDqQJzdGQ6Ol9fMjo6dGltZV9nZXQ8Y2hhciwgc3RkOjpfXzI6OmlzdHJlYW1idWZfaXRlcmF0b3I8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+Pj46Ol9fZ2V0X21vbnRobmFtZShpbnQmLCBzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+Jiwgc3RkOjpfXzI6OmlzdHJlYW1idWZfaXRlcmF0b3I8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+PiwgdW5zaWduZWQgaW50Jiwgc3RkOjpfXzI6OmN0eXBlPGNoYXI+IGNvbnN0JikgY29uc3TyA5oCc3RkOjpfXzI6OnRpbWVfZ2V0PGNoYXIsIHN0ZDo6X18yOjppc3RyZWFtYnVmX2l0ZXJhdG9yPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPj4+Ojpkb19nZXRfeWVhcihzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+LCBzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+LCBzdGQ6Ol9fMjo6aW9zX2Jhc2UmLCB1bnNpZ25lZCBpbnQmLCB0bSopIGNvbnN08wOkAnN0ZDo6X18yOjp0aW1lX2dldDxjaGFyLCBzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+Pjo6X19nZXRfeWVhcihpbnQmLCBzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+Jiwgc3RkOjpfXzI6OmlzdHJlYW1idWZfaXRlcmF0b3I8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+PiwgdW5zaWduZWQgaW50Jiwgc3RkOjpfXzI6OmN0eXBlPGNoYXI+IGNvbnN0JikgY29uc3T0A6ECaW50IHN0ZDo6X18yOjpfX2dldF91cF90b19uX2RpZ2l0czxjaGFyLCBzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+PihzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+Jiwgc3RkOjpfXzI6OmlzdHJlYW1idWZfaXRlcmF0b3I8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+PiwgdW5zaWduZWQgaW50Jiwgc3RkOjpfXzI6OmN0eXBlPGNoYXI+IGNvbnN0JiwgaW50KfUDoQJzdGQ6Ol9fMjo6dGltZV9nZXQ8Y2hhciwgc3RkOjpfXzI6OmlzdHJlYW1idWZfaXRlcmF0b3I8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+Pj46OmRvX2dldChzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+LCBzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+LCBzdGQ6Ol9fMjo6aW9zX2Jhc2UmLCB1bnNpZ25lZCBpbnQmLCB0bSosIGNoYXIsIGNoYXIpIGNvbnN09gPHAnN0ZDo6X18yOjp0aW1lX2dldDx3Y2hhcl90LCBzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjx3Y2hhcl90LCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8d2NoYXJfdD4+Pjo6Z2V0KHN0ZDo6X18yOjppc3RyZWFtYnVmX2l0ZXJhdG9yPHdjaGFyX3QsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czx3Y2hhcl90Pj4sIHN0ZDo6X18yOjppc3RyZWFtYnVmX2l0ZXJhdG9yPHdjaGFyX3QsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czx3Y2hhcl90Pj4sIHN0ZDo6X18yOjppb3NfYmFzZSYsIHVuc2lnbmVkIGludCYsIHRtKiwgd2NoYXJfdCBjb25zdCosIHdjaGFyX3QgY29uc3QqKSBjb25zdPcDrwJzdGQ6Ol9fMjo6dGltZV9nZXQ8d2NoYXJfdCwgc3RkOjpfXzI6OmlzdHJlYW1idWZfaXRlcmF0b3I8d2NoYXJfdCwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPHdjaGFyX3Q+Pj46OmRvX2dldF90aW1lKHN0ZDo6X18yOjppc3RyZWFtYnVmX2l0ZXJhdG9yPHdjaGFyX3QsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czx3Y2hhcl90Pj4sIHN0ZDo6X18yOjppc3RyZWFtYnVmX2l0ZXJhdG9yPHdjaGFyX3QsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czx3Y2hhcl90Pj4sIHN0ZDo6X18yOjppb3NfYmFzZSYsIHVuc2lnbmVkIGludCYsIHRtKikgY29uc3T4A68Cc3RkOjpfXzI6OnRpbWVfZ2V0PHdjaGFyX3QsIHN0ZDo6X18yOjppc3RyZWFtYnVmX2l0ZXJhdG9yPHdjaGFyX3QsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czx3Y2hhcl90Pj4+Ojpkb19nZXRfZGF0ZShzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjx3Y2hhcl90LCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8d2NoYXJfdD4+LCBzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjx3Y2hhcl90LCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8d2NoYXJfdD4+LCBzdGQ6Ol9fMjo6aW9zX2Jhc2UmLCB1bnNpZ25lZCBpbnQmLCB0bSopIGNvbnN0+QOyAnN0ZDo6X18yOjp0aW1lX2dldDx3Y2hhcl90LCBzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjx3Y2hhcl90LCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8d2NoYXJfdD4+Pjo6ZG9fZ2V0X3dlZWtkYXkoc3RkOjpfXzI6OmlzdHJlYW1idWZfaXRlcmF0b3I8d2NoYXJfdCwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPHdjaGFyX3Q+Piwgc3RkOjpfXzI6OmlzdHJlYW1idWZfaXRlcmF0b3I8d2NoYXJfdCwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPHdjaGFyX3Q+Piwgc3RkOjpfXzI6Omlvc19iYXNlJiwgdW5zaWduZWQgaW50JiwgdG0qKSBjb25zdPoDwwJzdGQ6Ol9fMjo6dGltZV9nZXQ8d2NoYXJfdCwgc3RkOjpfXzI6OmlzdHJlYW1idWZfaXRlcmF0b3I8d2NoYXJfdCwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPHdjaGFyX3Q+Pj46Ol9fZ2V0X3dlZWtkYXluYW1lKGludCYsIHN0ZDo6X18yOjppc3RyZWFtYnVmX2l0ZXJhdG9yPHdjaGFyX3QsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czx3Y2hhcl90Pj4mLCBzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjx3Y2hhcl90LCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8d2NoYXJfdD4+LCB1bnNpZ25lZCBpbnQmLCBzdGQ6Ol9fMjo6Y3R5cGU8d2NoYXJfdD4gY29uc3QmKSBjb25zdPsDtAJzdGQ6Ol9fMjo6dGltZV9nZXQ8d2NoYXJfdCwgc3RkOjpfXzI6OmlzdHJlYW1idWZfaXRlcmF0b3I8d2NoYXJfdCwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPHdjaGFyX3Q+Pj46OmRvX2dldF9tb250aG5hbWUoc3RkOjpfXzI6OmlzdHJlYW1idWZfaXRlcmF0b3I8d2NoYXJfdCwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPHdjaGFyX3Q+Piwgc3RkOjpfXzI6OmlzdHJlYW1idWZfaXRlcmF0b3I8d2NoYXJfdCwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPHdjaGFyX3Q+Piwgc3RkOjpfXzI6Omlvc19iYXNlJiwgdW5zaWduZWQgaW50JiwgdG0qKSBjb25zdPwDwQJzdGQ6Ol9fMjo6dGltZV9nZXQ8d2NoYXJfdCwgc3RkOjpfXzI6OmlzdHJlYW1idWZfaXRlcmF0b3I8d2NoYXJfdCwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPHdjaGFyX3Q+Pj46Ol9fZ2V0X21vbnRobmFtZShpbnQmLCBzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjx3Y2hhcl90LCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8d2NoYXJfdD4+Jiwgc3RkOjpfXzI6OmlzdHJlYW1idWZfaXRlcmF0b3I8d2NoYXJfdCwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPHdjaGFyX3Q+PiwgdW5zaWduZWQgaW50Jiwgc3RkOjpfXzI6OmN0eXBlPHdjaGFyX3Q+IGNvbnN0JikgY29uc3T9A68Cc3RkOjpfXzI6OnRpbWVfZ2V0PHdjaGFyX3QsIHN0ZDo6X18yOjppc3RyZWFtYnVmX2l0ZXJhdG9yPHdjaGFyX3QsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czx3Y2hhcl90Pj4+Ojpkb19nZXRfeWVhcihzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjx3Y2hhcl90LCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8d2NoYXJfdD4+LCBzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjx3Y2hhcl90LCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8d2NoYXJfdD4+LCBzdGQ6Ol9fMjo6aW9zX2Jhc2UmLCB1bnNpZ25lZCBpbnQmLCB0bSopIGNvbnN0/gO8AnN0ZDo6X18yOjp0aW1lX2dldDx3Y2hhcl90LCBzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjx3Y2hhcl90LCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8d2NoYXJfdD4+Pjo6X19nZXRfeWVhcihpbnQmLCBzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjx3Y2hhcl90LCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8d2NoYXJfdD4+Jiwgc3RkOjpfXzI6OmlzdHJlYW1idWZfaXRlcmF0b3I8d2NoYXJfdCwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPHdjaGFyX3Q+PiwgdW5zaWduZWQgaW50Jiwgc3RkOjpfXzI6OmN0eXBlPHdjaGFyX3Q+IGNvbnN0JikgY29uc3T/A7kCaW50IHN0ZDo6X18yOjpfX2dldF91cF90b19uX2RpZ2l0czx3Y2hhcl90LCBzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjx3Y2hhcl90LCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8d2NoYXJfdD4+PihzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjx3Y2hhcl90LCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8d2NoYXJfdD4+Jiwgc3RkOjpfXzI6OmlzdHJlYW1idWZfaXRlcmF0b3I8d2NoYXJfdCwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPHdjaGFyX3Q+PiwgdW5zaWduZWQgaW50Jiwgc3RkOjpfXzI6OmN0eXBlPHdjaGFyX3Q+IGNvbnN0JiwgaW50KYAEtgJzdGQ6Ol9fMjo6dGltZV9nZXQ8d2NoYXJfdCwgc3RkOjpfXzI6OmlzdHJlYW1idWZfaXRlcmF0b3I8d2NoYXJfdCwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPHdjaGFyX3Q+Pj46OmRvX2dldChzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjx3Y2hhcl90LCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8d2NoYXJfdD4+LCBzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjx3Y2hhcl90LCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8d2NoYXJfdD4+LCBzdGQ6Ol9fMjo6aW9zX2Jhc2UmLCB1bnNpZ25lZCBpbnQmLCB0bSosIGNoYXIsIGNoYXIpIGNvbnN0gQTcAXN0ZDo6X18yOjp0aW1lX3B1dDxjaGFyLCBzdGQ6Ol9fMjo6b3N0cmVhbWJ1Zl9pdGVyYXRvcjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+Pjo6ZG9fcHV0KHN0ZDo6X18yOjpvc3RyZWFtYnVmX2l0ZXJhdG9yPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPj4sIHN0ZDo6X18yOjppb3NfYmFzZSYsIGNoYXIsIHRtIGNvbnN0KiwgY2hhciwgY2hhcikgY29uc3SCBEpzdGQ6Ol9fMjo6X190aW1lX3B1dDo6X19kb19wdXQoY2hhciosIGNoYXIqJiwgdG0gY29uc3QqLCBjaGFyLCBjaGFyKSBjb25zdIME7gFzdGQ6Ol9fMjo6dGltZV9wdXQ8d2NoYXJfdCwgc3RkOjpfXzI6Om9zdHJlYW1idWZfaXRlcmF0b3I8d2NoYXJfdCwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPHdjaGFyX3Q+Pj46OmRvX3B1dChzdGQ6Ol9fMjo6b3N0cmVhbWJ1Zl9pdGVyYXRvcjx3Y2hhcl90LCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8d2NoYXJfdD4+LCBzdGQ6Ol9fMjo6aW9zX2Jhc2UmLCB3Y2hhcl90LCB0bSBjb25zdCosIGNoYXIsIGNoYXIpIGNvbnN0hAQsc3RkOjpfXzI6Ol9fdGhyb3dfcnVudGltZV9lcnJvcihjaGFyIGNvbnN0KimFBDtzdGQ6Ol9fMjo6bW9uZXlwdW5jdDxjaGFyLCBmYWxzZT46OmRvX2RlY2ltYWxfcG9pbnQoKSBjb25zdIYENnN0ZDo6X18yOjptb25leXB1bmN0PGNoYXIsIGZhbHNlPjo6ZG9fZ3JvdXBpbmcoKSBjb25zdIcEO3N0ZDo6X18yOjptb25leXB1bmN0PGNoYXIsIGZhbHNlPjo6ZG9fbmVnYXRpdmVfc2lnbigpIGNvbnN0iAQ4c3RkOjpfXzI6Om1vbmV5cHVuY3Q8Y2hhciwgZmFsc2U+Ojpkb19wb3NfZm9ybWF0KCkgY29uc3SJBD5zdGQ6Ol9fMjo6bW9uZXlwdW5jdDx3Y2hhcl90LCBmYWxzZT46OmRvX2RlY2ltYWxfcG9pbnQoKSBjb25zdIoEPnN0ZDo6X18yOjptb25leXB1bmN0PHdjaGFyX3QsIGZhbHNlPjo6ZG9fbmVnYXRpdmVfc2lnbigpIGNvbnN0iwS/AXN0ZDo6X18yOjpiYXNpY19zdHJpbmc8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPGNoYXI+Pjo6YmFzaWNfc3RyaW5nKHN0ZDo6X18yOjpiYXNpY19zdHJpbmc8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPGNoYXI+PiBjb25zdCYpjASlAnN0ZDo6X18yOjptb25leV9nZXQ8Y2hhciwgc3RkOjpfXzI6OmlzdHJlYW1idWZfaXRlcmF0b3I8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+Pj46OmRvX2dldChzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+LCBzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+LCBib29sLCBzdGQ6Ol9fMjo6aW9zX2Jhc2UmLCB1bnNpZ25lZCBpbnQmLCBsb25nIGRvdWJsZSYpIGNvbnN0jQSIA3N0ZDo6X18yOjptb25leV9nZXQ8Y2hhciwgc3RkOjpfXzI6OmlzdHJlYW1idWZfaXRlcmF0b3I8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+Pj46Ol9fZG9fZ2V0KHN0ZDo6X18yOjppc3RyZWFtYnVmX2l0ZXJhdG9yPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPj4mLCBzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+LCBib29sLCBzdGQ6Ol9fMjo6bG9jYWxlIGNvbnN0JiwgdW5zaWduZWQgaW50LCB1bnNpZ25lZCBpbnQmLCBib29sJiwgc3RkOjpfXzI6OmN0eXBlPGNoYXI+IGNvbnN0Jiwgc3RkOjpfXzI6OnVuaXF1ZV9wdHI8Y2hhciwgdm9pZCAoKikodm9pZCopPiYsIGNoYXIqJiwgY2hhciopjgRdc3RkOjpfXzI6OmlzdHJlYW1idWZfaXRlcmF0b3I8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+Pjo6b3BlcmF0b3IrK1thYmk6djE1MDA3XShpbnQpjwRmdm9pZCBzdGQ6Ol9fMjo6X19kb3VibGVfb3Jfbm90aGluZzxjaGFyPihzdGQ6Ol9fMjo6dW5pcXVlX3B0cjxjaGFyLCB2b2lkICgqKSh2b2lkKik+JiwgY2hhciomLCBjaGFyKiYpkASGAXZvaWQgc3RkOjpfXzI6Ol9fZG91YmxlX29yX25vdGhpbmc8dW5zaWduZWQgaW50PihzdGQ6Ol9fMjo6dW5pcXVlX3B0cjx1bnNpZ25lZCBpbnQsIHZvaWQgKCopKHZvaWQqKT4mLCB1bnNpZ25lZCBpbnQqJiwgdW5zaWduZWQgaW50KiYpkQTuAnN0ZDo6X18yOjptb25leV9nZXQ8Y2hhciwgc3RkOjpfXzI6OmlzdHJlYW1idWZfaXRlcmF0b3I8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+Pj46OmRvX2dldChzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+LCBzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+LCBib29sLCBzdGQ6Ol9fMjo6aW9zX2Jhc2UmLCB1bnNpZ25lZCBpbnQmLCBzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPiwgc3RkOjpfXzI6OmFsbG9jYXRvcjxjaGFyPj4mKSBjb25zdJIEmQJzdGQ6Ol9fMjo6ZW5hYmxlX2lmPF9faXNfY3BwMTdfZm9yd2FyZF9pdGVyYXRvcjxjaGFyKj46OnZhbHVlLCBzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPiwgc3RkOjpfXzI6OmFsbG9jYXRvcjxjaGFyPj4mPjo6dHlwZSBzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPiwgc3RkOjpfXzI6OmFsbG9jYXRvcjxjaGFyPj46OmFwcGVuZFthYmk6djE1MDA3XTxjaGFyKj4oY2hhciosIGNoYXIqKZMEe3N0ZDo6X18yOjpiYXNpY19zdHJpbmc8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPGNoYXI+Pjo6X19zZXRfc2l6ZVthYmk6djE1MDA3XSh1bnNpZ25lZCBsb25nKZQEggFzdGQ6Ol9fMjo6bW9uZXlwdW5jdDxjaGFyLCB0cnVlPiBjb25zdCYgc3RkOjpfXzI6OnVzZV9mYWNldFthYmk6djE1MDA3XTxzdGQ6Ol9fMjo6bW9uZXlwdW5jdDxjaGFyLCB0cnVlPj4oc3RkOjpfXzI6OmxvY2FsZSBjb25zdCYplQSEAXN0ZDo6X18yOjptb25leXB1bmN0PGNoYXIsIGZhbHNlPiBjb25zdCYgc3RkOjpfXzI6OnVzZV9mYWNldFthYmk6djE1MDA3XTxzdGQ6Ol9fMjo6bW9uZXlwdW5jdDxjaGFyLCBmYWxzZT4+KHN0ZDo6X18yOjpsb2NhbGUgY29uc3QmKZYEP3N0ZDo6X18yOjpfX3dyYXBfaXRlcjxjaGFyKj46Om9wZXJhdG9yK1thYmk6djE1MDA3XShsb25nKSBjb25zdJcEcXN0ZDo6X18yOjp1bmlxdWVfcHRyPGNoYXIsIHZvaWQgKCopKHZvaWQqKT46Om9wZXJhdG9yPVthYmk6djE1MDA3XShzdGQ6Ol9fMjo6dW5pcXVlX3B0cjxjaGFyLCB2b2lkICgqKSh2b2lkKik+JiYpmAS6AnN0ZDo6X18yOjptb25leV9nZXQ8d2NoYXJfdCwgc3RkOjpfXzI6OmlzdHJlYW1idWZfaXRlcmF0b3I8d2NoYXJfdCwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPHdjaGFyX3Q+Pj46OmRvX2dldChzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjx3Y2hhcl90LCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8d2NoYXJfdD4+LCBzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjx3Y2hhcl90LCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8d2NoYXJfdD4+LCBib29sLCBzdGQ6Ol9fMjo6aW9zX2Jhc2UmLCB1bnNpZ25lZCBpbnQmLCBsb25nIGRvdWJsZSYpIGNvbnN0mQSpA3N0ZDo6X18yOjptb25leV9nZXQ8d2NoYXJfdCwgc3RkOjpfXzI6OmlzdHJlYW1idWZfaXRlcmF0b3I8d2NoYXJfdCwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPHdjaGFyX3Q+Pj46Ol9fZG9fZ2V0KHN0ZDo6X18yOjppc3RyZWFtYnVmX2l0ZXJhdG9yPHdjaGFyX3QsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czx3Y2hhcl90Pj4mLCBzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjx3Y2hhcl90LCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8d2NoYXJfdD4+LCBib29sLCBzdGQ6Ol9fMjo6bG9jYWxlIGNvbnN0JiwgdW5zaWduZWQgaW50LCB1bnNpZ25lZCBpbnQmLCBib29sJiwgc3RkOjpfXzI6OmN0eXBlPHdjaGFyX3Q+IGNvbnN0Jiwgc3RkOjpfXzI6OnVuaXF1ZV9wdHI8d2NoYXJfdCwgdm9pZCAoKikodm9pZCopPiYsIHdjaGFyX3QqJiwgd2NoYXJfdCopmgRjc3RkOjpfXzI6OmlzdHJlYW1idWZfaXRlcmF0b3I8d2NoYXJfdCwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPHdjaGFyX3Q+Pjo6b3BlcmF0b3IrK1thYmk6djE1MDA3XShpbnQpmwSMA3N0ZDo6X18yOjptb25leV9nZXQ8d2NoYXJfdCwgc3RkOjpfXzI6OmlzdHJlYW1idWZfaXRlcmF0b3I8d2NoYXJfdCwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPHdjaGFyX3Q+Pj46OmRvX2dldChzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjx3Y2hhcl90LCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8d2NoYXJfdD4+LCBzdGQ6Ol9fMjo6aXN0cmVhbWJ1Zl9pdGVyYXRvcjx3Y2hhcl90LCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8d2NoYXJfdD4+LCBib29sLCBzdGQ6Ol9fMjo6aW9zX2Jhc2UmLCB1bnNpZ25lZCBpbnQmLCBzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPHdjaGFyX3QsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czx3Y2hhcl90Piwgc3RkOjpfXzI6OmFsbG9jYXRvcjx3Y2hhcl90Pj4mKSBjb25zdJwEtwJzdGQ6Ol9fMjo6ZW5hYmxlX2lmPF9faXNfY3BwMTdfZm9yd2FyZF9pdGVyYXRvcjx3Y2hhcl90Kj46OnZhbHVlLCBzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPHdjaGFyX3QsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czx3Y2hhcl90Piwgc3RkOjpfXzI6OmFsbG9jYXRvcjx3Y2hhcl90Pj4mPjo6dHlwZSBzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPHdjaGFyX3QsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czx3Y2hhcl90Piwgc3RkOjpfXzI6OmFsbG9jYXRvcjx3Y2hhcl90Pj46OmFwcGVuZFthYmk6djE1MDA3XTx3Y2hhcl90Kj4od2NoYXJfdCosIHdjaGFyX3QqKZ0EiAFzdGQ6Ol9fMjo6bW9uZXlwdW5jdDx3Y2hhcl90LCB0cnVlPiBjb25zdCYgc3RkOjpfXzI6OnVzZV9mYWNldFthYmk6djE1MDA3XTxzdGQ6Ol9fMjo6bW9uZXlwdW5jdDx3Y2hhcl90LCB0cnVlPj4oc3RkOjpfXzI6OmxvY2FsZSBjb25zdCYpngTVAXN0ZDo6X18yOjpiYXNpY19zdHJpbmc8d2NoYXJfdCwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPHdjaGFyX3Q+LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPHdjaGFyX3Q+Pjo6b3BlcmF0b3I9W2FiaTp2MTUwMDddKHN0ZDo6X18yOjpiYXNpY19zdHJpbmc8d2NoYXJfdCwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPHdjaGFyX3Q+LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPHdjaGFyX3Q+PiYmKZ8EigFzdGQ6Ol9fMjo6bW9uZXlwdW5jdDx3Y2hhcl90LCBmYWxzZT4gY29uc3QmIHN0ZDo6X18yOjp1c2VfZmFjZXRbYWJpOnYxNTAwN108c3RkOjpfXzI6Om1vbmV5cHVuY3Q8d2NoYXJfdCwgZmFsc2U+PihzdGQ6Ol9fMjo6bG9jYWxlIGNvbnN0JimgBEJzdGQ6Ol9fMjo6X193cmFwX2l0ZXI8d2NoYXJfdCo+OjpvcGVyYXRvcitbYWJpOnYxNTAwN10obG9uZykgY29uc3ShBNkBc3RkOjpfXzI6Om1vbmV5X3B1dDxjaGFyLCBzdGQ6Ol9fMjo6b3N0cmVhbWJ1Zl9pdGVyYXRvcjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+Pjo6ZG9fcHV0KHN0ZDo6X18yOjpvc3RyZWFtYnVmX2l0ZXJhdG9yPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPj4sIGJvb2wsIHN0ZDo6X18yOjppb3NfYmFzZSYsIGNoYXIsIGxvbmcgZG91YmxlKSBjb25zdKIEiANzdGQ6Ol9fMjo6X19tb25leV9wdXQ8Y2hhcj46Ol9fZ2F0aGVyX2luZm8oYm9vbCwgYm9vbCwgc3RkOjpfXzI6OmxvY2FsZSBjb25zdCYsIHN0ZDo6X18yOjptb25leV9iYXNlOjpwYXR0ZXJuJiwgY2hhciYsIGNoYXImLCBzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPiwgc3RkOjpfXzI6OmFsbG9jYXRvcjxjaGFyPj4mLCBzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPiwgc3RkOjpfXzI6OmFsbG9jYXRvcjxjaGFyPj4mLCBzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPiwgc3RkOjpfXzI6OmFsbG9jYXRvcjxjaGFyPj4mLCBpbnQmKaME1gNzdGQ6Ol9fMjo6X19tb25leV9wdXQ8Y2hhcj46Ol9fZm9ybWF0KGNoYXIqLCBjaGFyKiYsIGNoYXIqJiwgdW5zaWduZWQgaW50LCBjaGFyIGNvbnN0KiwgY2hhciBjb25zdCosIHN0ZDo6X18yOjpjdHlwZTxjaGFyPiBjb25zdCYsIGJvb2wsIHN0ZDo6X18yOjptb25leV9iYXNlOjpwYXR0ZXJuIGNvbnN0JiwgY2hhciwgY2hhciwgc3RkOjpfXzI6OmJhc2ljX3N0cmluZzxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4sIHN0ZDo6X18yOjphbGxvY2F0b3I8Y2hhcj4+IGNvbnN0Jiwgc3RkOjpfXzI6OmJhc2ljX3N0cmluZzxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4sIHN0ZDo6X18yOjphbGxvY2F0b3I8Y2hhcj4+IGNvbnN0Jiwgc3RkOjpfXzI6OmJhc2ljX3N0cmluZzxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4sIHN0ZDo6X18yOjphbGxvY2F0b3I8Y2hhcj4+IGNvbnN0JiwgaW50KaQEmgFjaGFyKiBzdGQ6Ol9fMjo6Y29weVthYmk6djE1MDA3XTxzdGQ6Ol9fMjo6X193cmFwX2l0ZXI8Y2hhciBjb25zdCo+LCBjaGFyKj4oc3RkOjpfXzI6Ol9fd3JhcF9pdGVyPGNoYXIgY29uc3QqPiwgc3RkOjpfXzI6Ol9fd3JhcF9pdGVyPGNoYXIgY29uc3QqPiwgY2hhcioppQSpAnN0ZDo6X18yOjptb25leV9wdXQ8Y2hhciwgc3RkOjpfXzI6Om9zdHJlYW1idWZfaXRlcmF0b3I8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+Pj46OmRvX3B1dChzdGQ6Ol9fMjo6b3N0cmVhbWJ1Zl9pdGVyYXRvcjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+LCBib29sLCBzdGQ6Ol9fMjo6aW9zX2Jhc2UmLCBjaGFyLCBzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPiwgc3RkOjpfXzI6OmFsbG9jYXRvcjxjaGFyPj4gY29uc3QmKSBjb25zdKYE6wFzdGQ6Ol9fMjo6bW9uZXlfcHV0PHdjaGFyX3QsIHN0ZDo6X18yOjpvc3RyZWFtYnVmX2l0ZXJhdG9yPHdjaGFyX3QsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czx3Y2hhcl90Pj4+Ojpkb19wdXQoc3RkOjpfXzI6Om9zdHJlYW1idWZfaXRlcmF0b3I8d2NoYXJfdCwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPHdjaGFyX3Q+PiwgYm9vbCwgc3RkOjpfXzI6Omlvc19iYXNlJiwgd2NoYXJfdCwgbG9uZyBkb3VibGUpIGNvbnN0pwSjA3N0ZDo6X18yOjpfX21vbmV5X3B1dDx3Y2hhcl90Pjo6X19nYXRoZXJfaW5mbyhib29sLCBib29sLCBzdGQ6Ol9fMjo6bG9jYWxlIGNvbnN0Jiwgc3RkOjpfXzI6Om1vbmV5X2Jhc2U6OnBhdHRlcm4mLCB3Y2hhcl90Jiwgd2NoYXJfdCYsIHN0ZDo6X18yOjpiYXNpY19zdHJpbmc8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPGNoYXI+PiYsIHN0ZDo6X18yOjpiYXNpY19zdHJpbmc8d2NoYXJfdCwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPHdjaGFyX3Q+LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPHdjaGFyX3Q+PiYsIHN0ZDo6X18yOjpiYXNpY19zdHJpbmc8d2NoYXJfdCwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPHdjaGFyX3Q+LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPHdjaGFyX3Q+PiYsIGludCYpqASDBHN0ZDo6X18yOjpfX21vbmV5X3B1dDx3Y2hhcl90Pjo6X19mb3JtYXQod2NoYXJfdCosIHdjaGFyX3QqJiwgd2NoYXJfdComLCB1bnNpZ25lZCBpbnQsIHdjaGFyX3QgY29uc3QqLCB3Y2hhcl90IGNvbnN0Kiwgc3RkOjpfXzI6OmN0eXBlPHdjaGFyX3Q+IGNvbnN0JiwgYm9vbCwgc3RkOjpfXzI6Om1vbmV5X2Jhc2U6OnBhdHRlcm4gY29uc3QmLCB3Y2hhcl90LCB3Y2hhcl90LCBzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPiwgc3RkOjpfXzI6OmFsbG9jYXRvcjxjaGFyPj4gY29uc3QmLCBzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPHdjaGFyX3QsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czx3Y2hhcl90Piwgc3RkOjpfXzI6OmFsbG9jYXRvcjx3Y2hhcl90Pj4gY29uc3QmLCBzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPHdjaGFyX3QsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czx3Y2hhcl90Piwgc3RkOjpfXzI6OmFsbG9jYXRvcjx3Y2hhcl90Pj4gY29uc3QmLCBpbnQpqQSsAXdjaGFyX3QqIHN0ZDo6X18yOjpjb3B5W2FiaTp2MTUwMDddPHN0ZDo6X18yOjpfX3dyYXBfaXRlcjx3Y2hhcl90IGNvbnN0Kj4sIHdjaGFyX3QqPihzdGQ6Ol9fMjo6X193cmFwX2l0ZXI8d2NoYXJfdCBjb25zdCo+LCBzdGQ6Ol9fMjo6X193cmFwX2l0ZXI8d2NoYXJfdCBjb25zdCo+LCB3Y2hhcl90KimqBMQCc3RkOjpfXzI6Om1vbmV5X3B1dDx3Y2hhcl90LCBzdGQ6Ol9fMjo6b3N0cmVhbWJ1Zl9pdGVyYXRvcjx3Y2hhcl90LCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8d2NoYXJfdD4+Pjo6ZG9fcHV0KHN0ZDo6X18yOjpvc3RyZWFtYnVmX2l0ZXJhdG9yPHdjaGFyX3QsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czx3Y2hhcl90Pj4sIGJvb2wsIHN0ZDo6X18yOjppb3NfYmFzZSYsIHdjaGFyX3QsIHN0ZDo6X18yOjpiYXNpY19zdHJpbmc8d2NoYXJfdCwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPHdjaGFyX3Q+LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPHdjaGFyX3Q+PiBjb25zdCYpIGNvbnN0qwSdAXN0ZDo6X18yOjptZXNzYWdlczxjaGFyPjo6ZG9fb3BlbihzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPiwgc3RkOjpfXzI6OmFsbG9jYXRvcjxjaGFyPj4gY29uc3QmLCBzdGQ6Ol9fMjo6bG9jYWxlIGNvbnN0JikgY29uc3SsBJMBc3RkOjpfXzI6Om1lc3NhZ2VzPGNoYXI+Ojpkb19nZXQobG9uZywgaW50LCBpbnQsIHN0ZDo6X18yOjpiYXNpY19zdHJpbmc8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPGNoYXI+PiBjb25zdCYpIGNvbnN0rQSfAXN0ZDo6X18yOjptZXNzYWdlczx3Y2hhcl90Pjo6ZG9fZ2V0KGxvbmcsIGludCwgaW50LCBzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPHdjaGFyX3QsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czx3Y2hhcl90Piwgc3RkOjpfXzI6OmFsbG9jYXRvcjx3Y2hhcl90Pj4gY29uc3QmKSBjb25zdK4EOXN0ZDo6X18yOjpjb2RlY3Z0PHdjaGFyX3QsIGNoYXIsIF9fbWJzdGF0ZV90Pjo6fmNvZGVjdnQoKa8ELXN0ZDo6X18yOjpsb2NhbGU6Ol9faW1wOjpfX2ltcCh1bnNpZ25lZCBsb25nKbAEfHN0ZDo6X18yOjp2ZWN0b3I8c3RkOjpfXzI6OmxvY2FsZTo6ZmFjZXQqLCBzdGQ6Ol9fMjo6X19zc29fYWxsb2NhdG9yPHN0ZDo6X18yOjpsb2NhbGU6OmZhY2V0KiwgMzB1bD4+Ojp+dmVjdG9yW2FiaTp2MTUwMDddKCmxBIgBc3RkOjpfXzI6OnZlY3RvcjxzdGQ6Ol9fMjo6bG9jYWxlOjpmYWNldCosIHN0ZDo6X18yOjpfX3Nzb19hbGxvY2F0b3I8c3RkOjpfXzI6OmxvY2FsZTo6ZmFjZXQqLCAzMHVsPj46Ol9fY29uc3RydWN0X2F0X2VuZCh1bnNpZ25lZCBsb25nKbIErgFzdGQ6Ol9fMjo6X190cmFuc2FjdGlvbjxzdGQ6Ol9fMjo6dmVjdG9yPHN0ZDo6X18yOjpsb2NhbGU6OmZhY2V0Kiwgc3RkOjpfXzI6Ol9fc3NvX2FsbG9jYXRvcjxzdGQ6Ol9fMjo6bG9jYWxlOjpmYWNldCosIDMwdWw+Pjo6X19kZXN0cm95X3ZlY3Rvcj46On5fX3RyYW5zYWN0aW9uW2FiaTp2MTUwMDddKCmzBHxzdGQ6Ol9fMjo6dmVjdG9yPHN0ZDo6X18yOjpsb2NhbGU6OmZhY2V0Kiwgc3RkOjpfXzI6Ol9fc3NvX2FsbG9jYXRvcjxzdGQ6Ol9fMjo6bG9jYWxlOjpmYWNldCosIDMwdWw+Pjo6X19jbGVhclthYmk6djE1MDA3XSgptAQdc3RkOjpfXzI6OmxvY2FsZTo6aWQ6Ol9fZ2V0KCm1BEBzdGQ6Ol9fMjo6bG9jYWxlOjpfX2ltcDo6aW5zdGFsbChzdGQ6Ol9fMjo6bG9jYWxlOjpmYWNldCosIGxvbmcptgSRAXN0ZDo6X18yOjp2ZWN0b3I8c3RkOjpfXzI6OmxvY2FsZTo6ZmFjZXQqLCBzdGQ6Ol9fMjo6X19zc29fYWxsb2NhdG9yPHN0ZDo6X18yOjpsb2NhbGU6OmZhY2V0KiwgMzB1bD4+OjpfX2Rlc3Ryb3lfdmVjdG9yOjpvcGVyYXRvcigpW2FiaTp2MTUwMDddKCm3BIQBc3RkOjpfXzI6OnVuaXF1ZV9wdHI8c3RkOjpfXzI6OmxvY2FsZTo6ZmFjZXQsIHN0ZDo6X18yOjooYW5vbnltb3VzIG5hbWVzcGFjZSk6OnJlbGVhc2U+OjpyZXNldFthYmk6djE1MDA3XShzdGQ6Ol9fMjo6bG9jYWxlOjpmYWNldCopuAQhc3RkOjpfXzI6OmxvY2FsZTo6X19pbXA6On5fX2ltcCgpuQQjc3RkOjpfXzI6OmxvY2FsZTo6X19pbXA6On5fX2ltcCgpLjG6BH5zdGQ6Ol9fMjo6dmVjdG9yPHN0ZDo6X18yOjpsb2NhbGU6OmZhY2V0Kiwgc3RkOjpfXzI6Ol9fc3NvX2FsbG9jYXRvcjxzdGQ6Ol9fMjo6bG9jYWxlOjpmYWNldCosIDMwdWw+Pjo6X19hcHBlbmQodW5zaWduZWQgbG9uZym7BC5zdGQ6Ol9fMjo6bG9jYWxlOjpfX2ltcDo6aGFzX2ZhY2V0KGxvbmcpIGNvbnN0vAQac3RkOjpfXzI6OmxvY2FsZTo6bG9jYWxlKCm9BB5zdGQ6Ol9fMjo6bG9jYWxlOjppZDo6X19pbml0KCm+BCtzdGQ6Ol9fMjo6bG9jYWxlOjpmYWNldDo6X19vbl96ZXJvX3NoYXJlZCgpvwR0dm9pZCBzdGQ6Ol9fMjo6X19jYWxsX29uY2VfcHJveHlbYWJpOnYxNTAwN108c3RkOjpfXzI6OnR1cGxlPHN0ZDo6X18yOjooYW5vbnltb3VzIG5hbWVzcGFjZSk6Ol9fZmFrZV9iaW5kJiY+Pih2b2lkKinABD1zdGQ6Ol9fMjo6Y3R5cGU8d2NoYXJfdD46OmRvX2lzKHVuc2lnbmVkIGxvbmcsIHdjaGFyX3QpIGNvbnN0wQRVc3RkOjpfXzI6OmN0eXBlPHdjaGFyX3Q+Ojpkb19pcyh3Y2hhcl90IGNvbnN0Kiwgd2NoYXJfdCBjb25zdCosIHVuc2lnbmVkIGxvbmcqKSBjb25zdMIEWXN0ZDo6X18yOjpjdHlwZTx3Y2hhcl90Pjo6ZG9fc2Nhbl9pcyh1bnNpZ25lZCBsb25nLCB3Y2hhcl90IGNvbnN0Kiwgd2NoYXJfdCBjb25zdCopIGNvbnN0wwRac3RkOjpfXzI6OmN0eXBlPHdjaGFyX3Q+Ojpkb19zY2FuX25vdCh1bnNpZ25lZCBsb25nLCB3Y2hhcl90IGNvbnN0Kiwgd2NoYXJfdCBjb25zdCopIGNvbnN0xAQzc3RkOjpfXzI6OmN0eXBlPHdjaGFyX3Q+Ojpkb190b3VwcGVyKHdjaGFyX3QpIGNvbnN0xQREc3RkOjpfXzI6OmN0eXBlPHdjaGFyX3Q+Ojpkb190b3VwcGVyKHdjaGFyX3QqLCB3Y2hhcl90IGNvbnN0KikgY29uc3TGBDNzdGQ6Ol9fMjo6Y3R5cGU8d2NoYXJfdD46OmRvX3RvbG93ZXIod2NoYXJfdCkgY29uc3THBERzdGQ6Ol9fMjo6Y3R5cGU8d2NoYXJfdD46OmRvX3RvbG93ZXIod2NoYXJfdCosIHdjaGFyX3QgY29uc3QqKSBjb25zdMgETHN0ZDo6X18yOjpjdHlwZTx3Y2hhcl90Pjo6ZG9fd2lkZW4oY2hhciBjb25zdCosIGNoYXIgY29uc3QqLCB3Y2hhcl90KikgY29uc3TJBDhzdGQ6Ol9fMjo6Y3R5cGU8d2NoYXJfdD46OmRvX25hcnJvdyh3Y2hhcl90LCBjaGFyKSBjb25zdMoEVnN0ZDo6X18yOjpjdHlwZTx3Y2hhcl90Pjo6ZG9fbmFycm93KHdjaGFyX3QgY29uc3QqLCB3Y2hhcl90IGNvbnN0KiwgY2hhciwgY2hhciopIGNvbnN0ywQfc3RkOjpfXzI6OmN0eXBlPGNoYXI+Ojp+Y3R5cGUoKcwEIXN0ZDo6X18yOjpjdHlwZTxjaGFyPjo6fmN0eXBlKCkuMc0ELXN0ZDo6X18yOjpjdHlwZTxjaGFyPjo6ZG9fdG91cHBlcihjaGFyKSBjb25zdM4EO3N0ZDo6X18yOjpjdHlwZTxjaGFyPjo6ZG9fdG91cHBlcihjaGFyKiwgY2hhciBjb25zdCopIGNvbnN0zwQtc3RkOjpfXzI6OmN0eXBlPGNoYXI+Ojpkb190b2xvd2VyKGNoYXIpIGNvbnN00AQ7c3RkOjpfXzI6OmN0eXBlPGNoYXI+Ojpkb190b2xvd2VyKGNoYXIqLCBjaGFyIGNvbnN0KikgY29uc3TRBEZzdGQ6Ol9fMjo6Y3R5cGU8Y2hhcj46OmRvX3dpZGVuKGNoYXIgY29uc3QqLCBjaGFyIGNvbnN0KiwgY2hhciopIGNvbnN00gQyc3RkOjpfXzI6OmN0eXBlPGNoYXI+Ojpkb19uYXJyb3coY2hhciwgY2hhcikgY29uc3TTBE1zdGQ6Ol9fMjo6Y3R5cGU8Y2hhcj46OmRvX25hcnJvdyhjaGFyIGNvbnN0KiwgY2hhciBjb25zdCosIGNoYXIsIGNoYXIqKSBjb25zdNQEhAFzdGQ6Ol9fMjo6Y29kZWN2dDxjaGFyLCBjaGFyLCBfX21ic3RhdGVfdD46OmRvX291dChfX21ic3RhdGVfdCYsIGNoYXIgY29uc3QqLCBjaGFyIGNvbnN0KiwgY2hhciBjb25zdComLCBjaGFyKiwgY2hhciosIGNoYXIqJikgY29uc3TVBGBzdGQ6Ol9fMjo6Y29kZWN2dDxjaGFyLCBjaGFyLCBfX21ic3RhdGVfdD46OmRvX3Vuc2hpZnQoX19tYnN0YXRlX3QmLCBjaGFyKiwgY2hhciosIGNoYXIqJikgY29uc3TWBHJzdGQ6Ol9fMjo6Y29kZWN2dDxjaGFyLCBjaGFyLCBfX21ic3RhdGVfdD46OmRvX2xlbmd0aChfX21ic3RhdGVfdCYsIGNoYXIgY29uc3QqLCBjaGFyIGNvbnN0KiwgdW5zaWduZWQgbG9uZykgY29uc3TXBDtzdGQ6Ol9fMjo6Y29kZWN2dDx3Y2hhcl90LCBjaGFyLCBfX21ic3RhdGVfdD46On5jb2RlY3Z0KCkuMdgEkAFzdGQ6Ol9fMjo6Y29kZWN2dDx3Y2hhcl90LCBjaGFyLCBfX21ic3RhdGVfdD46OmRvX291dChfX21ic3RhdGVfdCYsIHdjaGFyX3QgY29uc3QqLCB3Y2hhcl90IGNvbnN0Kiwgd2NoYXJfdCBjb25zdComLCBjaGFyKiwgY2hhciosIGNoYXIqJikgY29uc3TZBFhzdGQ6Ol9fMjo6X19saWJjcHBfd2NydG9tYl9sW2FiaTp2MTUwMDddKGNoYXIqLCB3Y2hhcl90LCBfX21ic3RhdGVfdCosIF9fbG9jYWxlX3N0cnVjdCop2gSPAXN0ZDo6X18yOjpjb2RlY3Z0PHdjaGFyX3QsIGNoYXIsIF9fbWJzdGF0ZV90Pjo6ZG9faW4oX19tYnN0YXRlX3QmLCBjaGFyIGNvbnN0KiwgY2hhciBjb25zdCosIGNoYXIgY29uc3QqJiwgd2NoYXJfdCosIHdjaGFyX3QqLCB3Y2hhcl90KiYpIGNvbnN02wRuc3RkOjpfXzI6Ol9fbGliY3BwX21icnRvd2NfbFthYmk6djE1MDA3XSh3Y2hhcl90KiwgY2hhciBjb25zdCosIHVuc2lnbmVkIGxvbmcsIF9fbWJzdGF0ZV90KiwgX19sb2NhbGVfc3RydWN0KincBGNzdGQ6Ol9fMjo6Y29kZWN2dDx3Y2hhcl90LCBjaGFyLCBfX21ic3RhdGVfdD46OmRvX3Vuc2hpZnQoX19tYnN0YXRlX3QmLCBjaGFyKiwgY2hhciosIGNoYXIqJikgY29uc3TdBEJzdGQ6Ol9fMjo6Y29kZWN2dDx3Y2hhcl90LCBjaGFyLCBfX21ic3RhdGVfdD46OmRvX2VuY29kaW5nKCkgY29uc3TeBD1zdGQ6Ol9fMjo6X19saWJjcHBfbWJfY3VyX21heF9sW2FiaTp2MTUwMDddKF9fbG9jYWxlX3N0cnVjdCop3wR1c3RkOjpfXzI6OmNvZGVjdnQ8d2NoYXJfdCwgY2hhciwgX19tYnN0YXRlX3Q+Ojpkb19sZW5ndGgoX19tYnN0YXRlX3QmLCBjaGFyIGNvbnN0KiwgY2hhciBjb25zdCosIHVuc2lnbmVkIGxvbmcpIGNvbnN04AREc3RkOjpfXzI6OmNvZGVjdnQ8d2NoYXJfdCwgY2hhciwgX19tYnN0YXRlX3Q+Ojpkb19tYXhfbGVuZ3RoKCkgY29uc3ThBJQBc3RkOjpfXzI6OmNvZGVjdnQ8Y2hhcjE2X3QsIGNoYXIsIF9fbWJzdGF0ZV90Pjo6ZG9fb3V0KF9fbWJzdGF0ZV90JiwgY2hhcjE2X3QgY29uc3QqLCBjaGFyMTZfdCBjb25zdCosIGNoYXIxNl90IGNvbnN0KiYsIGNoYXIqLCBjaGFyKiwgY2hhciomKSBjb25zdOIEkwFzdGQ6Ol9fMjo6Y29kZWN2dDxjaGFyMTZfdCwgY2hhciwgX19tYnN0YXRlX3Q+Ojpkb19pbihfX21ic3RhdGVfdCYsIGNoYXIgY29uc3QqLCBjaGFyIGNvbnN0KiwgY2hhciBjb25zdComLCBjaGFyMTZfdCosIGNoYXIxNl90KiwgY2hhcjE2X3QqJikgY29uc3TjBHZzdGQ6Ol9fMjo6Y29kZWN2dDxjaGFyMTZfdCwgY2hhciwgX19tYnN0YXRlX3Q+Ojpkb19sZW5ndGgoX19tYnN0YXRlX3QmLCBjaGFyIGNvbnN0KiwgY2hhciBjb25zdCosIHVuc2lnbmVkIGxvbmcpIGNvbnN05ARFc3RkOjpfXzI6OmNvZGVjdnQ8Y2hhcjE2X3QsIGNoYXIsIF9fbWJzdGF0ZV90Pjo6ZG9fbWF4X2xlbmd0aCgpIGNvbnN05QSUAXN0ZDo6X18yOjpjb2RlY3Z0PGNoYXIzMl90LCBjaGFyLCBfX21ic3RhdGVfdD46OmRvX291dChfX21ic3RhdGVfdCYsIGNoYXIzMl90IGNvbnN0KiwgY2hhcjMyX3QgY29uc3QqLCBjaGFyMzJfdCBjb25zdComLCBjaGFyKiwgY2hhciosIGNoYXIqJikgY29uc3TmBJMBc3RkOjpfXzI6OmNvZGVjdnQ8Y2hhcjMyX3QsIGNoYXIsIF9fbWJzdGF0ZV90Pjo6ZG9faW4oX19tYnN0YXRlX3QmLCBjaGFyIGNvbnN0KiwgY2hhciBjb25zdCosIGNoYXIgY29uc3QqJiwgY2hhcjMyX3QqLCBjaGFyMzJfdCosIGNoYXIzMl90KiYpIGNvbnN05wR2c3RkOjpfXzI6OmNvZGVjdnQ8Y2hhcjMyX3QsIGNoYXIsIF9fbWJzdGF0ZV90Pjo6ZG9fbGVuZ3RoKF9fbWJzdGF0ZV90JiwgY2hhciBjb25zdCosIGNoYXIgY29uc3QqLCB1bnNpZ25lZCBsb25nKSBjb25zdOgEJXN0ZDo6X18yOjpudW1wdW5jdDxjaGFyPjo6fm51bXB1bmN0KCnpBCdzdGQ6Ol9fMjo6bnVtcHVuY3Q8Y2hhcj46On5udW1wdW5jdCgpLjHqBChzdGQ6Ol9fMjo6bnVtcHVuY3Q8d2NoYXJfdD46On5udW1wdW5jdCgp6wQqc3RkOjpfXzI6Om51bXB1bmN0PHdjaGFyX3Q+Ojp+bnVtcHVuY3QoKS4x7AQyc3RkOjpfXzI6Om51bXB1bmN0PGNoYXI+Ojpkb19kZWNpbWFsX3BvaW50KCkgY29uc3TtBDJzdGQ6Ol9fMjo6bnVtcHVuY3Q8Y2hhcj46OmRvX3Rob3VzYW5kc19zZXAoKSBjb25zdO4ELXN0ZDo6X18yOjpudW1wdW5jdDxjaGFyPjo6ZG9fZ3JvdXBpbmcoKSBjb25zdO8EMHN0ZDo6X18yOjpudW1wdW5jdDx3Y2hhcl90Pjo6ZG9fZ3JvdXBpbmcoKSBjb25zdPAELXN0ZDo6X18yOjpudW1wdW5jdDxjaGFyPjo6ZG9fdHJ1ZW5hbWUoKSBjb25zdPEEMHN0ZDo6X18yOjpudW1wdW5jdDx3Y2hhcl90Pjo6ZG9fdHJ1ZW5hbWUoKSBjb25zdPIElwFzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPHdjaGFyX3QsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czx3Y2hhcl90Piwgc3RkOjpfXzI6OmFsbG9jYXRvcjx3Y2hhcl90Pj46OmJhc2ljX3N0cmluZ1thYmk6djE1MDA3XTxzdGQ6Om51bGxwdHJfdD4od2NoYXJfdCBjb25zdCop8wQuc3RkOjpfXzI6Om51bXB1bmN0PGNoYXI+Ojpkb19mYWxzZW5hbWUoKSBjb25zdPQEMXN0ZDo6X18yOjpudW1wdW5jdDx3Y2hhcl90Pjo6ZG9fZmFsc2VuYW1lKCkgY29uc3T1BHhzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPiwgc3RkOjpfXzI6OmFsbG9jYXRvcjxjaGFyPj46Om9wZXJhdG9yPVthYmk6djE1MDA3XShjaGFyIGNvbnN0Kin2BDVzdGQ6Ol9fMjo6X190aW1lX2dldF9jX3N0b3JhZ2U8Y2hhcj46Ol9fd2Vla3MoKSBjb25zdPcEGl9fY3h4X2dsb2JhbF9hcnJheV9kdG9yLjU3+AQ4c3RkOjpfXzI6Ol9fdGltZV9nZXRfY19zdG9yYWdlPHdjaGFyX3Q+OjpfX3dlZWtzKCkgY29uc3T5BBpfX2N4eF9nbG9iYWxfYXJyYXlfZHRvci43MvoEhAFzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPHdjaGFyX3QsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czx3Y2hhcl90Piwgc3RkOjpfXzI6OmFsbG9jYXRvcjx3Y2hhcl90Pj46Om9wZXJhdG9yPVthYmk6djE1MDA3XSh3Y2hhcl90IGNvbnN0Kin7BDZzdGQ6Ol9fMjo6X190aW1lX2dldF9jX3N0b3JhZ2U8Y2hhcj46Ol9fbW9udGhzKCkgY29uc3T8BBpfX2N4eF9nbG9iYWxfYXJyYXlfZHRvci44N/0EOXN0ZDo6X18yOjpfX3RpbWVfZ2V0X2Nfc3RvcmFnZTx3Y2hhcl90Pjo6X19tb250aHMoKSBjb25zdP4EG19fY3h4X2dsb2JhbF9hcnJheV9kdG9yLjExMf8ENXN0ZDo6X18yOjpfX3RpbWVfZ2V0X2Nfc3RvcmFnZTxjaGFyPjo6X19hbV9wbSgpIGNvbnN0gAUbX19jeHhfZ2xvYmFsX2FycmF5X2R0b3IuMTM1gQU4c3RkOjpfXzI6Ol9fdGltZV9nZXRfY19zdG9yYWdlPHdjaGFyX3Q+OjpfX2FtX3BtKCkgY29uc3SCBRtfX2N4eF9nbG9iYWxfYXJyYXlfZHRvci4xMziDBTFzdGQ6Ol9fMjo6X190aW1lX2dldF9jX3N0b3JhZ2U8Y2hhcj46Ol9feCgpIGNvbnN0hAUZX19jeHhfZ2xvYmFsX2FycmF5X2R0b3IuM4UFNHN0ZDo6X18yOjpfX3RpbWVfZ2V0X2Nfc3RvcmFnZTx3Y2hhcl90Pjo6X194KCkgY29uc3SGBRpfX2N4eF9nbG9iYWxfYXJyYXlfZHRvci4zMocFMXN0ZDo6X18yOjpfX3RpbWVfZ2V0X2Nfc3RvcmFnZTxjaGFyPjo6X19YKCkgY29uc3SIBRpfX2N4eF9nbG9iYWxfYXJyYXlfZHRvci4zNIkFNHN0ZDo6X18yOjpfX3RpbWVfZ2V0X2Nfc3RvcmFnZTx3Y2hhcl90Pjo6X19YKCkgY29uc3SKBRpfX2N4eF9nbG9iYWxfYXJyYXlfZHRvci4zNosFMXN0ZDo6X18yOjpfX3RpbWVfZ2V0X2Nfc3RvcmFnZTxjaGFyPjo6X19jKCkgY29uc3SMBRpfX2N4eF9nbG9iYWxfYXJyYXlfZHRvci4zOI0FNHN0ZDo6X18yOjpfX3RpbWVfZ2V0X2Nfc3RvcmFnZTx3Y2hhcl90Pjo6X19jKCkgY29uc3SOBRpfX2N4eF9nbG9iYWxfYXJyYXlfZHRvci40MI8FMXN0ZDo6X18yOjpfX3RpbWVfZ2V0X2Nfc3RvcmFnZTxjaGFyPjo6X19yKCkgY29uc3SQBRpfX2N4eF9nbG9iYWxfYXJyYXlfZHRvci40MpEFNHN0ZDo6X18yOjpfX3RpbWVfZ2V0X2Nfc3RvcmFnZTx3Y2hhcl90Pjo6X19yKCkgY29uc3SSBRpfX2N4eF9nbG9iYWxfYXJyYXlfZHRvci40NJMFc3N0ZDo6X18yOjp0aW1lX3B1dDxjaGFyLCBzdGQ6Ol9fMjo6b3N0cmVhbWJ1Zl9pdGVyYXRvcjxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4+Pjo6fnRpbWVfcHV0W2FiaTp2MTUwMDddKCmUBXVzdGQ6Ol9fMjo6dGltZV9wdXQ8Y2hhciwgc3RkOjpfXzI6Om9zdHJlYW1idWZfaXRlcmF0b3I8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+Pj46On50aW1lX3B1dFthYmk6djE1MDA3XSgpLjGVBdYBc3RkOjpfXzI6Ol9fYWxsb2NhdGlvbl9yZXN1bHQ8c3RkOjpfXzI6OmFsbG9jYXRvcl90cmFpdHM8c3RkOjpfXzI6OmFsbG9jYXRvcjx3Y2hhcl90Pj46OnBvaW50ZXI+IHN0ZDo6X18yOjpfX2FsbG9jYXRlX2F0X2xlYXN0W2FiaTp2MTUwMDddPHN0ZDo6X18yOjphbGxvY2F0b3I8d2NoYXJfdD4+KHN0ZDo6X18yOjphbGxvY2F0b3I8d2NoYXJfdD4mLCB1bnNpZ25lZCBsb25nKZYFQXN0ZDo6X18yOjphbGxvY2F0b3I8d2NoYXJfdD46OmFsbG9jYXRlW2FiaTp2MTUwMDddKHVuc2lnbmVkIGxvbmcplwV/c3RkOjpfXzI6OmJhc2ljX3N0cmluZzxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4sIHN0ZDo6X18yOjphbGxvY2F0b3I8Y2hhcj4+OjpfX2VyYXNlX3RvX2VuZFthYmk6djE1MDA3XSh1bnNpZ25lZCBsb25nKZgFiwFzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPiwgc3RkOjpfXzI6OmFsbG9jYXRvcjxjaGFyPj46Ol9fbnVsbF90ZXJtaW5hdGVfYXRbYWJpOnYxNTAwN10oY2hhciosIHVuc2lnbmVkIGxvbmcpmQWIAXN0ZDo6X18yOjphbGxvY2F0b3JfdHJhaXRzPHN0ZDo6X18yOjphbGxvY2F0b3I8d2NoYXJfdD4+OjpkZWFsbG9jYXRlW2FiaTp2MTUwMDddKHN0ZDo6X18yOjphbGxvY2F0b3I8d2NoYXJfdD4mLCB3Y2hhcl90KiwgdW5zaWduZWQgbG9uZymaBU1zdGQ6Ol9fMjo6YWxsb2NhdG9yPHdjaGFyX3Q+OjpkZWFsbG9jYXRlW2FiaTp2MTUwMDddKHdjaGFyX3QqLCB1bnNpZ25lZCBsb25nKZsFuQFhdXRvIHN0ZDo6X18yOjpfX3Vud3JhcF9yYW5nZVthYmk6djE1MDA3XTxzdGQ6Ol9fMjo6X193cmFwX2l0ZXI8Y2hhciBjb25zdCo+LCBzdGQ6Ol9fMjo6X193cmFwX2l0ZXI8Y2hhciBjb25zdCo+PihzdGQ6Ol9fMjo6X193cmFwX2l0ZXI8Y2hhciBjb25zdCo+LCBzdGQ6Ol9fMjo6X193cmFwX2l0ZXI8Y2hhciBjb25zdCo+KZwFxQJkZWNsdHlwZShzdGQ6Ol9fMjo6X191bndyYXBfaXRlcl9pbXBsPHN0ZDo6X18yOjpfX3dyYXBfaXRlcjxjaGFyIGNvbnN0Kj4sIHRydWU+OjpfX3Vud3JhcChzdGQ6OmRlY2x2YWw8c3RkOjpfXzI6Ol9fd3JhcF9pdGVyPGNoYXIgY29uc3QqPj4oKSkpIHN0ZDo6X18yOjpfX3Vud3JhcF9pdGVyW2FiaTp2MTUwMDddPHN0ZDo6X18yOjpfX3dyYXBfaXRlcjxjaGFyIGNvbnN0Kj4sIHN0ZDo6X18yOjpfX3Vud3JhcF9pdGVyX2ltcGw8c3RkOjpfXzI6Ol9fd3JhcF9pdGVyPGNoYXIgY29uc3QqPiwgdHJ1ZT4sIDA+KHN0ZDo6X18yOjpfX3dyYXBfaXRlcjxjaGFyIGNvbnN0Kj4pnQV3c3RkOjpfXzI6OnZlY3RvcjxzdGQ6Ol9fMjo6bG9jYWxlOjpmYWNldCosIHN0ZDo6X18yOjpfX3Nzb19hbGxvY2F0b3I8c3RkOjpfXzI6OmxvY2FsZTo6ZmFjZXQqLCAzMHVsPj46Om1heF9zaXplKCkgY29uc3SeBa0Cc3RkOjpfXzI6Ol9fYWxsb2NhdGlvbl9yZXN1bHQ8c3RkOjpfXzI6OmFsbG9jYXRvcl90cmFpdHM8c3RkOjpfXzI6Ol9fc3NvX2FsbG9jYXRvcjxzdGQ6Ol9fMjo6bG9jYWxlOjpmYWNldCosIDMwdWw+Pjo6cG9pbnRlcj4gc3RkOjpfXzI6Ol9fYWxsb2NhdGVfYXRfbGVhc3RbYWJpOnYxNTAwN108c3RkOjpfXzI6Ol9fc3NvX2FsbG9jYXRvcjxzdGQ6Ol9fMjo6bG9jYWxlOjpmYWNldCosIDMwdWw+PihzdGQ6Ol9fMjo6X19zc29fYWxsb2NhdG9yPHN0ZDo6X18yOjpsb2NhbGU6OmZhY2V0KiwgMzB1bD4mLCB1bnNpZ25lZCBsb25nKZ8F0wFzdGQ6Ol9fMjo6YWxsb2NhdG9yX3RyYWl0czxzdGQ6Ol9fMjo6X19zc29fYWxsb2NhdG9yPHN0ZDo6X18yOjpsb2NhbGU6OmZhY2V0KiwgMzB1bD4+OjpkZWFsbG9jYXRlW2FiaTp2MTUwMDddKHN0ZDo6X18yOjpfX3Nzb19hbGxvY2F0b3I8c3RkOjpfXzI6OmxvY2FsZTo6ZmFjZXQqLCAzMHVsPiYsIHN0ZDo6X18yOjpsb2NhbGU6OmZhY2V0KiosIHVuc2lnbmVkIGxvbmcpoAWkAXN0ZDo6X18yOjp2ZWN0b3I8c3RkOjpfXzI6OmxvY2FsZTo6ZmFjZXQqLCBzdGQ6Ol9fMjo6X19zc29fYWxsb2NhdG9yPHN0ZDo6X18yOjpsb2NhbGU6OmZhY2V0KiwgMzB1bD4+OjpfX2Jhc2VfZGVzdHJ1Y3RfYXRfZW5kW2FiaTp2MTUwMDddKHN0ZDo6X18yOjpsb2NhbGU6OmZhY2V0KiopoQWBAXN0ZDo6X18yOjpfX3NwbGl0X2J1ZmZlcjxzdGQ6Ol9fMjo6bG9jYWxlOjpmYWNldCosIHN0ZDo6X18yOjpfX3Nzb19hbGxvY2F0b3I8c3RkOjpfXzI6OmxvY2FsZTo6ZmFjZXQqLCAzMHVsPiY+Ojp+X19zcGxpdF9idWZmZXIoKaIFN3N0ZDo6X18yOjpfX3Rocm93X291dF9vZl9yYW5nZVthYmk6djE1MDA3XShjaGFyIGNvbnN0KimjBS5zdGQ6Ol9fMjo6X190aW1lX3B1dDo6X190aW1lX3B1dFthYmk6djE1MDA3XSgppAUJc3RydG9sbF9spQUKc3RydG91bGxfbKYFLXN0ZDo6X18yOjpfX3NoYXJlZF9jb3VudDo6fl9fc2hhcmVkX2NvdW50KCkuMacFGHN0ZDo6X190aHJvd19iYWRfYWxsb2MoKagFG29wZXJhdG9yIG5ldyh1bnNpZ25lZCBsb25nKakFSnN0ZDo6X18yOjpfX2xpYmNwcF9hbGlnbmVkX2FsbG9jW2FiaTp2MTUwMDddKHVuc2lnbmVkIGxvbmcsIHVuc2lnbmVkIGxvbmcpqgVBc3RkOjpfXzI6Ol9fZnM6OmZpbGVzeXN0ZW06OmZpbGVzeXN0ZW1fZXJyb3I6On5maWxlc3lzdGVtX2Vycm9yKCmrBWdzdGQ6Ol9fMjo6c2hhcmVkX3B0cjxzdGQ6Ol9fMjo6X19mczo6ZmlsZXN5c3RlbTo6ZmlsZXN5c3RlbV9lcnJvcjo6X1N0b3JhZ2U+Ojp+c2hhcmVkX3B0clthYmk6djE1MDA3XSgprAVDc3RkOjpfXzI6Ol9fZnM6OmZpbGVzeXN0ZW06OmZpbGVzeXN0ZW1fZXJyb3I6On5maWxlc3lzdGVtX2Vycm9yKCkuMa0FQHN0ZDo6X18yOjpfX2ZzOjpmaWxlc3lzdGVtOjpmaWxlc3lzdGVtX2Vycm9yOjpfX2NyZWF0ZV93aGF0KGludCmuBVpzdGQ6Ol9fMjo6X19mczo6ZmlsZXN5c3RlbTo6ZGV0YWlsOjooYW5vbnltb3VzIG5hbWVzcGFjZSk6OmZvcm1hdF9zdHJpbmcoY2hhciBjb25zdCosIC4uLimvBSxzdGQ6Ol9fMjo6X19saWJjcHBfdW5yZWFjaGFibGVbYWJpOnYxNTAwN10oKbAFNXN0ZDo6X18yOjpfX2ZzOjpmaWxlc3lzdGVtOjpwYXRoOjp+cGF0aFthYmk6djE1MDA3XSgpsQVbc3RkOjpfXzI6Ol9fZnM6OmZpbGVzeXN0ZW06OnBhdGg6OnBhdGhbYWJpOnYxNTAwN10oc3RkOjpfXzI6Ol9fZnM6OmZpbGVzeXN0ZW06OnBhdGggY29uc3QmKbIFlwJzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPiwgc3RkOjpfXzI6OmFsbG9jYXRvcjxjaGFyPj4gc3RkOjpfXzI6Om9wZXJhdG9yK1thYmk6djE1MDA3XTxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4sIHN0ZDo6X18yOjphbGxvY2F0b3I8Y2hhcj4+KHN0ZDo6X18yOjpiYXNpY19zdHJpbmc8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPGNoYXI+PiYmLCBjaGFyIGNvbnN0KimzBcUBc3RkOjpfXzI6OmJhc2ljX3N0cmluZzxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4sIHN0ZDo6X18yOjphbGxvY2F0b3I8Y2hhcj4+OjphcHBlbmRbYWJpOnYxNTAwN10oc3RkOjpfXzI6OmJhc2ljX3N0cmluZzxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4sIHN0ZDo6X18yOjphbGxvY2F0b3I8Y2hhcj4+IGNvbnN0Jim0BUZzdGQ6Ol9fMjo6X19mczo6ZmlsZXN5c3RlbTo6ZmlsZXN5c3RlbV9lcnJvcjo6d2hhdFthYmk6djE1MDA3XSgpIGNvbnN0tQVhc3RkOjpfXzI6Ol9fZnM6OmZpbGVzeXN0ZW06OmRldGFpbDo6KGFub255bW91cyBuYW1lc3BhY2UpOjpmb3JtYXRfc3RyaW5nX2ltcGwoY2hhciBjb25zdCosIHZvaWQqKbYFkQFzdGQ6Ol9fMjo6X19mczo6ZmlsZXN5c3RlbTo6ZGV0YWlsOjooYW5vbnltb3VzIG5hbWVzcGFjZSk6OkVycm9ySGFuZGxlcjx2b2lkPjo6cmVwb3J0X2ltcGwoc3RkOjpfXzI6OmVycm9yX2NvZGUgY29uc3QmLCBjaGFyIGNvbnN0Kiwgdm9pZCopIGNvbnN0twW4AnN0ZDo6X18yOjpzaGFyZWRfcHRyPHN0ZDo6X18yOjpfX2ZzOjpmaWxlc3lzdGVtOjpmaWxlc3lzdGVtX2Vycm9yOjpfU3RvcmFnZT4gc3RkOjpfXzI6Om1ha2Vfc2hhcmVkW2FiaTp2MTUwMDddPHN0ZDo6X18yOjpfX2ZzOjpmaWxlc3lzdGVtOjpmaWxlc3lzdGVtX2Vycm9yOjpfU3RvcmFnZSwgc3RkOjpfXzI6Ol9fZnM6OmZpbGVzeXN0ZW06OnBhdGgsIHN0ZDo6X18yOjpfX2ZzOjpmaWxlc3lzdGVtOjpwYXRoLCB2b2lkPihzdGQ6Ol9fMjo6X19mczo6ZmlsZXN5c3RlbTo6cGF0aCYmLCBzdGQ6Ol9fMjo6X19mczo6ZmlsZXN5c3RlbTo6cGF0aCYmKbgF3QNzdGQ6Ol9fMjo6c2hhcmVkX3B0cjxzdGQ6Ol9fMjo6X19mczo6ZmlsZXN5c3RlbTo6ZmlsZXN5c3RlbV9lcnJvcjo6X1N0b3JhZ2U+IHN0ZDo6X18yOjphbGxvY2F0ZV9zaGFyZWRbYWJpOnYxNTAwN108c3RkOjpfXzI6Ol9fZnM6OmZpbGVzeXN0ZW06OmZpbGVzeXN0ZW1fZXJyb3I6Ol9TdG9yYWdlLCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPHN0ZDo6X18yOjpfX2ZzOjpmaWxlc3lzdGVtOjpmaWxlc3lzdGVtX2Vycm9yOjpfU3RvcmFnZT4sIHN0ZDo6X18yOjpfX2ZzOjpmaWxlc3lzdGVtOjpwYXRoLCBzdGQ6Ol9fMjo6X19mczo6ZmlsZXN5c3RlbTo6cGF0aCwgdm9pZD4oc3RkOjpfXzI6OmFsbG9jYXRvcjxzdGQ6Ol9fMjo6X19mczo6ZmlsZXN5c3RlbTo6ZmlsZXN5c3RlbV9lcnJvcjo6X1N0b3JhZ2U+IGNvbnN0Jiwgc3RkOjpfXzI6Ol9fZnM6OmZpbGVzeXN0ZW06OnBhdGgmJiwgc3RkOjpfXzI6Ol9fZnM6OmZpbGVzeXN0ZW06OnBhdGgmJim5BfkBc3RkOjpfXzI6Ol9fYWxsb2NhdGlvbl9ndWFyZDxzdGQ6Ol9fMjo6YWxsb2NhdG9yPHN0ZDo6X18yOjpfX3NoYXJlZF9wdHJfZW1wbGFjZTxzdGQ6Ol9fMjo6X19mczo6ZmlsZXN5c3RlbTo6ZmlsZXN5c3RlbV9lcnJvcjo6X1N0b3JhZ2UsIHN0ZDo6X18yOjphbGxvY2F0b3I8c3RkOjpfXzI6Ol9fZnM6OmZpbGVzeXN0ZW06OmZpbGVzeXN0ZW1fZXJyb3I6Ol9TdG9yYWdlPj4+Pjo6fl9fYWxsb2NhdGlvbl9ndWFyZFthYmk6djE1MDA3XSgpugW8AXN0ZDo6X18yOjpfX3NoYXJlZF9wdHJfZW1wbGFjZTxzdGQ6Ol9fMjo6X19mczo6ZmlsZXN5c3RlbTo6ZmlsZXN5c3RlbV9lcnJvcjo6X1N0b3JhZ2UsIHN0ZDo6X18yOjphbGxvY2F0b3I8c3RkOjpfXzI6Ol9fZnM6OmZpbGVzeXN0ZW06OmZpbGVzeXN0ZW1fZXJyb3I6Ol9TdG9yYWdlPj46On5fX3NoYXJlZF9wdHJfZW1wbGFjZSgpuwW+AXN0ZDo6X18yOjpfX3NoYXJlZF9wdHJfZW1wbGFjZTxzdGQ6Ol9fMjo6X19mczo6ZmlsZXN5c3RlbTo6ZmlsZXN5c3RlbV9lcnJvcjo6X1N0b3JhZ2UsIHN0ZDo6X18yOjphbGxvY2F0b3I8c3RkOjpfXzI6Ol9fZnM6OmZpbGVzeXN0ZW06OmZpbGVzeXN0ZW1fZXJyb3I6Ol9TdG9yYWdlPj46On5fX3NoYXJlZF9wdHJfZW1wbGFjZSgpLjG8BbcBc3RkOjpfXzI6Ol9fc2hhcmVkX3B0cl9lbXBsYWNlPHN0ZDo6X18yOjpfX2ZzOjpmaWxlc3lzdGVtOjpmaWxlc3lzdGVtX2Vycm9yOjpfU3RvcmFnZSwgc3RkOjpfXzI6OmFsbG9jYXRvcjxzdGQ6Ol9fMjo6X19mczo6ZmlsZXN5c3RlbTo6ZmlsZXN5c3RlbV9lcnJvcjo6X1N0b3JhZ2U+Pjo6X19vbl96ZXJvX3NoYXJlZCgpvQW8AXN0ZDo6X18yOjpfX3NoYXJlZF9wdHJfZW1wbGFjZTxzdGQ6Ol9fMjo6X19mczo6ZmlsZXN5c3RlbTo6ZmlsZXN5c3RlbV9lcnJvcjo6X1N0b3JhZ2UsIHN0ZDo6X18yOjphbGxvY2F0b3I8c3RkOjpfXzI6Ol9fZnM6OmZpbGVzeXN0ZW06OmZpbGVzeXN0ZW1fZXJyb3I6Ol9TdG9yYWdlPj46Ol9fb25femVyb19zaGFyZWRfd2VhaygpvgWeAXN0ZDo6X18yOjpfX2ZzOjpmaWxlc3lzdGVtOjpmaWxlc3lzdGVtX2Vycm9yOjpfU3RvcmFnZTo6X1N0b3JhZ2VbYWJpOnYxNTAwN10oc3RkOjpfXzI6Ol9fZnM6OmZpbGVzeXN0ZW06OnBhdGggY29uc3QmLCBzdGQ6Ol9fMjo6X19mczo6ZmlsZXN5c3RlbTo6cGF0aCBjb25zdCYpvwWFA3N0ZDo6X18yOjphbGxvY2F0b3I8c3RkOjpfXzI6Ol9fc2hhcmVkX3B0cl9lbXBsYWNlPHN0ZDo6X18yOjpfX2ZzOjpmaWxlc3lzdGVtOjpmaWxlc3lzdGVtX2Vycm9yOjpfU3RvcmFnZSwgc3RkOjpfXzI6OmFsbG9jYXRvcjxzdGQ6Ol9fMjo6X19mczo6ZmlsZXN5c3RlbTo6ZmlsZXN5c3RlbV9lcnJvcjo6X1N0b3JhZ2U+Pj46OmRlYWxsb2NhdGVbYWJpOnYxNTAwN10oc3RkOjpfXzI6Ol9fc2hhcmVkX3B0cl9lbXBsYWNlPHN0ZDo6X18yOjpfX2ZzOjpmaWxlc3lzdGVtOjpmaWxlc3lzdGVtX2Vycm9yOjpfU3RvcmFnZSwgc3RkOjpfXzI6OmFsbG9jYXRvcjxzdGQ6Ol9fMjo6X19mczo6ZmlsZXN5c3RlbTo6ZmlsZXN5c3RlbV9lcnJvcjo6X1N0b3JhZ2U+PiosIHVuc2lnbmVkIGxvbmcpwAV6c3RkOjpsb2dpY19lcnJvcjo6bG9naWNfZXJyb3Ioc3RkOjpfXzI6OmJhc2ljX3N0cmluZzxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4sIHN0ZDo6X18yOjphbGxvY2F0b3I8Y2hhcj4+IGNvbnN0JinBBT1zdGQ6Ol9fMjo6X19saWJjcHBfcmVmc3RyaW5nOjpfX2xpYmNwcF9yZWZzdHJpbmcoY2hhciBjb25zdCopwgUqc3RkOjpsb2dpY19lcnJvcjo6bG9naWNfZXJyb3IoY2hhciBjb25zdCopwwV+c3RkOjpydW50aW1lX2Vycm9yOjpydW50aW1lX2Vycm9yKHN0ZDo6X18yOjpiYXNpY19zdHJpbmc8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPGNoYXI+PiBjb25zdCYpxAUuc3RkOjpydW50aW1lX2Vycm9yOjpydW50aW1lX2Vycm9yKGNoYXIgY29uc3QqKcUFRHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPjo6bW92ZShjaGFyKiwgY2hhciBjb25zdCosIHVuc2lnbmVkIGxvbmcpxgXSAXN0ZDo6X18yOjpiYXNpY19zdHJpbmc8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPGNoYXI+Pjo6X19ncm93X2J5X2FuZF9yZXBsYWNlKHVuc2lnbmVkIGxvbmcsIHVuc2lnbmVkIGxvbmcsIHVuc2lnbmVkIGxvbmcsIHVuc2lnbmVkIGxvbmcsIHVuc2lnbmVkIGxvbmcsIHVuc2lnbmVkIGxvbmcsIGNoYXIgY29uc3QqKccFZXN0ZDo6X18yOjpiYXNpY19zdHJpbmc8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPGNoYXI+Pjo6fmJhc2ljX3N0cmluZygpyAW5AXN0ZDo6X18yOjpiYXNpY19zdHJpbmc8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPGNoYXI+Pjo6X19ncm93X2J5KHVuc2lnbmVkIGxvbmcsIHVuc2lnbmVkIGxvbmcsIHVuc2lnbmVkIGxvbmcsIHVuc2lnbmVkIGxvbmcsIHVuc2lnbmVkIGxvbmcsIHVuc2lnbmVkIGxvbmcpyQU/c3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+Ojphc3NpZ24oY2hhciosIHVuc2lnbmVkIGxvbmcsIGNoYXIpygWDAXN0ZDo6X18yOjpiYXNpY19zdHJpbmc8Y2hhciwgc3RkOjpfXzI6OmNoYXJfdHJhaXRzPGNoYXI+LCBzdGQ6Ol9fMjo6YWxsb2NhdG9yPGNoYXI+Pjo6X19hc3NpZ25fZXh0ZXJuYWwoY2hhciBjb25zdCosIHVuc2lnbmVkIGxvbmcpywV4c3RkOjpfXzI6OmJhc2ljX3N0cmluZzxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4sIHN0ZDo6X18yOjphbGxvY2F0b3I8Y2hhcj4+OjphcHBlbmQoY2hhciBjb25zdCosIHVuc2lnbmVkIGxvbmcpzAV4c3RkOjpfXzI6OmJhc2ljX3N0cmluZzxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4sIHN0ZDo6X18yOjphbGxvY2F0b3I8Y2hhcj4+OjppbnNlcnQodW5zaWduZWQgbG9uZywgY2hhciBjb25zdCopzQVlc3RkOjpfXzI6OmJhc2ljX3N0cmluZzxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4sIHN0ZDo6X18yOjphbGxvY2F0b3I8Y2hhcj4+OjpwdXNoX2JhY2soY2hhcinOBWlzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPiwgc3RkOjpfXzI6OmFsbG9jYXRvcjxjaGFyPj46OmFwcGVuZChjaGFyIGNvbnN0KinPBd4Bc3RkOjpfXzI6OmJhc2ljX3N0cmluZzx3Y2hhcl90LCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8d2NoYXJfdD4sIHN0ZDo6X18yOjphbGxvY2F0b3I8d2NoYXJfdD4+OjpfX2dyb3dfYnlfYW5kX3JlcGxhY2UodW5zaWduZWQgbG9uZywgdW5zaWduZWQgbG9uZywgdW5zaWduZWQgbG9uZywgdW5zaWduZWQgbG9uZywgdW5zaWduZWQgbG9uZywgdW5zaWduZWQgbG9uZywgd2NoYXJfdCBjb25zdCop0AVuc3RkOjpfXzI6OmJhc2ljX3N0cmluZzx3Y2hhcl90LCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8d2NoYXJfdD4sIHN0ZDo6X18yOjphbGxvY2F0b3I8d2NoYXJfdD4+Ojp+YmFzaWNfc3RyaW5nKCnRBcIBc3RkOjpfXzI6OmJhc2ljX3N0cmluZzx3Y2hhcl90LCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8d2NoYXJfdD4sIHN0ZDo6X18yOjphbGxvY2F0b3I8d2NoYXJfdD4+OjpfX2dyb3dfYnkodW5zaWduZWQgbG9uZywgdW5zaWduZWQgbG9uZywgdW5zaWduZWQgbG9uZywgdW5zaWduZWQgbG9uZywgdW5zaWduZWQgbG9uZywgdW5zaWduZWQgbG9uZynSBXFzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPHdjaGFyX3QsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czx3Y2hhcl90Piwgc3RkOjpfXzI6OmFsbG9jYXRvcjx3Y2hhcl90Pj46OnB1c2hfYmFjayh3Y2hhcl90KdMFkAJzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPiwgc3RkOjpfXzI6OmFsbG9jYXRvcjxjaGFyPj4gc3RkOjpfXzI6Om9wZXJhdG9yKzxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4sIHN0ZDo6X18yOjphbGxvY2F0b3I8Y2hhcj4+KGNoYXIgY29uc3QqLCBzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPiwgc3RkOjpfXzI6OmFsbG9jYXRvcjxjaGFyPj4gY29uc3QmKdQFIXN0ZDo6X18yOjp0b19zdHJpbmcodW5zaWduZWQgaW50KdUFPHN0ZDo6X18yOjpfX2l0b2E6Ol9fYXBwZW5kMlthYmk6djE1MDA3XShjaGFyKiwgdW5zaWduZWQgaW50KdYFPHN0ZDo6X18yOjpfX2l0b2E6Ol9fYXBwZW5kNFthYmk6djE1MDA3XShjaGFyKiwgdW5zaWduZWQgaW50KdcFPHN0ZDo6X18yOjpfX2l0b2E6Ol9fYXBwZW5kNlthYmk6djE1MDA3XShjaGFyKiwgdW5zaWduZWQgaW50KdgFPHN0ZDo6X18yOjpfX2l0b2E6Ol9fYXBwZW5kOFthYmk6djE1MDA3XShjaGFyKiwgdW5zaWduZWQgaW50KdkFPHN0ZDo6X18yOjplcnJvcl9jYXRlZ29yeTo6ZGVmYXVsdF9lcnJvcl9jb25kaXRpb24oaW50KSBjb25zdNoFUXN0ZDo6X18yOjplcnJvcl9jYXRlZ29yeTo6ZXF1aXZhbGVudChpbnQsIHN0ZDo6X18yOjplcnJvcl9jb25kaXRpb24gY29uc3QmKSBjb25zdNsFTHN0ZDo6X18yOjplcnJvcl9jYXRlZ29yeTo6ZXF1aXZhbGVudChzdGQ6Ol9fMjo6ZXJyb3JfY29kZSBjb25zdCYsIGludCkgY29uc3TcBSpzdGQ6Ol9fMjo6X19kb19tZXNzYWdlOjptZXNzYWdlKGludCkgY29uc3TdBTBzdGQ6Ol9fMjo6X19nZW5lcmljX2Vycm9yX2NhdGVnb3J5OjpuYW1lKCkgY29uc3TeBTZzdGQ6Ol9fMjo6X19nZW5lcmljX2Vycm9yX2NhdGVnb3J5OjptZXNzYWdlKGludCkgY29uc3TfBS9zdGQ6Ol9fMjo6X19zeXN0ZW1fZXJyb3JfY2F0ZWdvcnk6Om5hbWUoKSBjb25zdOAFRXN0ZDo6X18yOjpfX3N5c3RlbV9lcnJvcl9jYXRlZ29yeTo6ZGVmYXVsdF9lcnJvcl9jb25kaXRpb24oaW50KSBjb25zdOEFkQFzdGQ6Ol9fMjo6c3lzdGVtX2Vycm9yOjpfX2luaXQoc3RkOjpfXzI6OmVycm9yX2NvZGUgY29uc3QmLCBzdGQ6Ol9fMjo6YmFzaWNfc3RyaW5nPGNoYXIsIHN0ZDo6X18yOjpjaGFyX3RyYWl0czxjaGFyPiwgc3RkOjpfXzI6OmFsbG9jYXRvcjxjaGFyPj4p4gWXAXN0ZDo6X18yOjpzeXN0ZW1fZXJyb3I6OnN5c3RlbV9lcnJvcihzdGQ6Ol9fMjo6ZXJyb3JfY29kZSwgc3RkOjpfXzI6OmJhc2ljX3N0cmluZzxjaGFyLCBzdGQ6Ol9fMjo6Y2hhcl90cmFpdHM8Y2hhcj4sIHN0ZDo6X18yOjphbGxvY2F0b3I8Y2hhcj4+IGNvbnN0JinjBUdzdGQ6Ol9fMjo6c3lzdGVtX2Vycm9yOjpzeXN0ZW1fZXJyb3Ioc3RkOjpfXzI6OmVycm9yX2NvZGUsIGNoYXIgY29uc3QqKeQFGF9fY3hhX2FsbG9jYXRlX2V4Y2VwdGlvbuUFFF9fY3hhX2ZyZWVfZXhjZXB0aW9u5gULX19jeGFfdGhyb3fnBUtfX2N4eGFiaXYxOjpleGNlcHRpb25fY2xlYW51cF9mdW5jKF9VbndpbmRfUmVhc29uX0NvZGUsIF9VbndpbmRfRXhjZXB0aW9uKinoBSJfX2N4YV9kZWNyZW1lbnRfZXhjZXB0aW9uX3JlZmNvdW506QURX19jeGFfYmVnaW5fY2F0Y2jqBQ9fX2N4YV9lbmRfY2F0Y2jrBQ1fX2N4YV9yZXRocm937AUiX19jeGFfaW5jcmVtZW50X2V4Y2VwdGlvbl9yZWZjb3VudO0FDWFib3J0X21lc3NhZ2XuBR5kZW1hbmdsaW5nX3Rlcm1pbmF0ZV9oYW5kbGVyKCnvBR9kZW1hbmdsaW5nX3VuZXhwZWN0ZWRfaGFuZGxlcigp8AUQc3RkOjp0ZXJtaW5hdGUoKfEFHHN0ZDo6X190ZXJtaW5hdGUodm9pZCAoKikoKSnyBRJfX2N4YV9wdXJlX3ZpcnR1YWzzBS9fX2N4eGFiaXYxOjpfX2FsaWduZWRfZnJlZV93aXRoX2ZhbGxiYWNrKHZvaWQqKfQFYV9fY3h4YWJpdjE6Ol9fZnVuZGFtZW50YWxfdHlwZV9pbmZvOjpjYW5fY2F0Y2goX19jeHhhYml2MTo6X19zaGltX3R5cGVfaW5mbyBjb25zdCosIHZvaWQqJikgY29uc3T1BTxpc19lcXVhbChzdGQ6OnR5cGVfaW5mbyBjb25zdCosIHN0ZDo6dHlwZV9pbmZvIGNvbnN0KiwgYm9vbCn2BVtfX2N4eGFiaXYxOjpfX2NsYXNzX3R5cGVfaW5mbzo6Y2FuX2NhdGNoKF9fY3h4YWJpdjE6Ol9fc2hpbV90eXBlX2luZm8gY29uc3QqLCB2b2lkKiYpIGNvbnN09wUOX19keW5hbWljX2Nhc3T4BWtfX2N4eGFiaXYxOjpfX2NsYXNzX3R5cGVfaW5mbzo6cHJvY2Vzc19mb3VuZF9iYXNlX2NsYXNzKF9fY3h4YWJpdjE6Ol9fZHluYW1pY19jYXN0X2luZm8qLCB2b2lkKiwgaW50KSBjb25zdPkFbl9fY3h4YWJpdjE6Ol9fY2xhc3NfdHlwZV9pbmZvOjpoYXNfdW5hbWJpZ3VvdXNfcHVibGljX2Jhc2UoX19jeHhhYml2MTo6X19keW5hbWljX2Nhc3RfaW5mbyosIHZvaWQqLCBpbnQpIGNvbnN0+gVxX19jeHhhYml2MTo6X19zaV9jbGFzc190eXBlX2luZm86Omhhc191bmFtYmlndW91c19wdWJsaWNfYmFzZShfX2N4eGFiaXYxOjpfX2R5bmFtaWNfY2FzdF9pbmZvKiwgdm9pZCosIGludCkgY29uc3T7BXNfX2N4eGFiaXYxOjpfX2Jhc2VfY2xhc3NfdHlwZV9pbmZvOjpoYXNfdW5hbWJpZ3VvdXNfcHVibGljX2Jhc2UoX19jeHhhYml2MTo6X19keW5hbWljX2Nhc3RfaW5mbyosIHZvaWQqLCBpbnQpIGNvbnN0/AVyX19jeHhhYml2MTo6X192bWlfY2xhc3NfdHlwZV9pbmZvOjpoYXNfdW5hbWJpZ3VvdXNfcHVibGljX2Jhc2UoX19jeHhhYml2MTo6X19keW5hbWljX2Nhc3RfaW5mbyosIHZvaWQqLCBpbnQpIGNvbnN0/QVdX19jeHhhYml2MTo6X19wb2ludGVyX3R5cGVfaW5mbzo6Y2FuX2NhdGNoKF9fY3h4YWJpdjE6Ol9fc2hpbV90eXBlX2luZm8gY29uc3QqLCB2b2lkKiYpIGNvbnN0/gVmX19jeHhhYml2MTo6X19wb2ludGVyX3RvX21lbWJlcl90eXBlX2luZm86OmNhbl9jYXRjaF9uZXN0ZWQoX19jeHhhYml2MTo6X19zaGltX3R5cGVfaW5mbyBjb25zdCopIGNvbnN0/wWDAV9fY3h4YWJpdjE6Ol9fY2xhc3NfdHlwZV9pbmZvOjpwcm9jZXNzX3N0YXRpY190eXBlX2Fib3ZlX2RzdChfX2N4eGFiaXYxOjpfX2R5bmFtaWNfY2FzdF9pbmZvKiwgdm9pZCBjb25zdCosIHZvaWQgY29uc3QqLCBpbnQpIGNvbnN0gAZzX19jeHhhYml2MTo6X192bWlfY2xhc3NfdHlwZV9pbmZvOjpzZWFyY2hfYmVsb3dfZHN0KF9fY3h4YWJpdjE6Ol9fZHluYW1pY19jYXN0X2luZm8qLCB2b2lkIGNvbnN0KiwgaW50LCBib29sKSBjb25zdIEGgQFfX2N4eGFiaXYxOjpfX2Jhc2VfY2xhc3NfdHlwZV9pbmZvOjpzZWFyY2hfYWJvdmVfZHN0KF9fY3h4YWJpdjE6Ol9fZHluYW1pY19jYXN0X2luZm8qLCB2b2lkIGNvbnN0Kiwgdm9pZCBjb25zdCosIGludCwgYm9vbCkgY29uc3SCBnRfX2N4eGFiaXYxOjpfX2Jhc2VfY2xhc3NfdHlwZV9pbmZvOjpzZWFyY2hfYmVsb3dfZHN0KF9fY3h4YWJpdjE6Ol9fZHluYW1pY19jYXN0X2luZm8qLCB2b2lkIGNvbnN0KiwgaW50LCBib29sKSBjb25zdIMGcl9fY3h4YWJpdjE6Ol9fc2lfY2xhc3NfdHlwZV9pbmZvOjpzZWFyY2hfYmVsb3dfZHN0KF9fY3h4YWJpdjE6Ol9fZHluYW1pY19jYXN0X2luZm8qLCB2b2lkIGNvbnN0KiwgaW50LCBib29sKSBjb25zdIQGb19fY3h4YWJpdjE6Ol9fY2xhc3NfdHlwZV9pbmZvOjpzZWFyY2hfYmVsb3dfZHN0KF9fY3h4YWJpdjE6Ol9fZHluYW1pY19jYXN0X2luZm8qLCB2b2lkIGNvbnN0KiwgaW50LCBib29sKSBjb25zdIUGgAFfX2N4eGFiaXYxOjpfX3ZtaV9jbGFzc190eXBlX2luZm86OnNlYXJjaF9hYm92ZV9kc3QoX19jeHhhYml2MTo6X19keW5hbWljX2Nhc3RfaW5mbyosIHZvaWQgY29uc3QqLCB2b2lkIGNvbnN0KiwgaW50LCBib29sKSBjb25zdIYGf19fY3h4YWJpdjE6Ol9fc2lfY2xhc3NfdHlwZV9pbmZvOjpzZWFyY2hfYWJvdmVfZHN0KF9fY3h4YWJpdjE6Ol9fZHluYW1pY19jYXN0X2luZm8qLCB2b2lkIGNvbnN0Kiwgdm9pZCBjb25zdCosIGludCwgYm9vbCkgY29uc3SHBnxfX2N4eGFiaXYxOjpfX2NsYXNzX3R5cGVfaW5mbzo6c2VhcmNoX2Fib3ZlX2RzdChfX2N4eGFiaXYxOjpfX2R5bmFtaWNfY2FzdF9pbmZvKiwgdm9pZCBjb25zdCosIHZvaWQgY29uc3QqLCBpbnQsIGJvb2wpIGNvbnN0iAYcc3RkOjpleGNlcHRpb246OndoYXQoKSBjb25zdIkGHHN0ZDo6YmFkX2FsbG9jOjp3aGF0KCkgY29uc3SKBidzdGQ6OmJhZF9hcnJheV9uZXdfbGVuZ3RoOjp3aGF0KCkgY29uc3SLBiBzdGQ6OmxvZ2ljX2Vycm9yOjp+bG9naWNfZXJyb3IoKYwGM3N0ZDo6X18yOjpfX2xpYmNwcF9yZWZzdHJpbmc6On5fX2xpYmNwcF9yZWZzdHJpbmcoKY0GInN0ZDo6bG9naWNfZXJyb3I6On5sb2dpY19lcnJvcigpLjGOBiRzdGQ6OnJ1bnRpbWVfZXJyb3I6On5ydW50aW1lX2Vycm9yKCmPBhtzdGQ6OmJhZF9jYXN0Ojp3aGF0KCkgY29uc3SQBlNfX2N4eGFiaXYxOjpyZWFkRW5jb2RlZFBvaW50ZXIodW5zaWduZWQgY2hhciBjb25zdCoqLCB1bnNpZ25lZCBjaGFyLCB1bnNpZ25lZCBsb25nKZEGLl9fY3h4YWJpdjE6OnJlYWRVTEVCMTI4KHVuc2lnbmVkIGNoYXIgY29uc3QqKimSBi5fX2N4eGFiaXYxOjpyZWFkU0xFQjEyOCh1bnNpZ25lZCBjaGFyIGNvbnN0KiopkwaAAV9fY3h4YWJpdjE6OmdldF9zaGltX3R5cGVfaW5mbyh1bnNpZ25lZCBsb25nIGxvbmcsIHVuc2lnbmVkIGNoYXIgY29uc3QqLCB1bnNpZ25lZCBjaGFyLCBib29sLCBfVW53aW5kX0V4Y2VwdGlvbiosIHVuc2lnbmVkIGxvbmcplAY0X19jeHhhYml2MTo6Y2FsbF90ZXJtaW5hdGUoYm9vbCwgX1Vud2luZF9FeGNlcHRpb24qKZUGF19VbndpbmRfQ2FsbFBlcnNvbmFsaXR5lgYVZW1zY3JpcHRlbl9zdGFja19pbml0lwYZZW1zY3JpcHRlbl9zdGFja19nZXRfZnJlZZgGGWVtc2NyaXB0ZW5fc3RhY2tfZ2V0X2Jhc2WZBhhlbXNjcmlwdGVuX3N0YWNrX2dldF9lbmSaBg5fX2N4YV9kZW1hbmdsZZsG5QEoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OkFic3RyYWN0TWFuZ2xpbmdQYXJzZXI8KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpNYW5nbGluZ1BhcnNlcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+LCAoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+Ojp+QWJzdHJhY3RNYW5nbGluZ1BhcnNlcigpnAZHKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpPdXRwdXRCdWZmZXI6Om9wZXJhdG9yKz0oY2hhcimdBkwoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OlN0cmluZ1ZpZXc6OlN0cmluZ1ZpZXcoY2hhciBjb25zdCopngaKAihhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6QWJzdHJhY3RNYW5nbGluZ1BhcnNlcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok1hbmdsaW5nUGFyc2VyPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6RGVmYXVsdEFsbG9jYXRvcj4sIChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6RGVmYXVsdEFsbG9jYXRvcj46OmNvbnN1bWVJZigoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OlN0cmluZ1ZpZXcpnwbbAShhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6QWJzdHJhY3RNYW5nbGluZ1BhcnNlcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok1hbmdsaW5nUGFyc2VyPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6RGVmYXVsdEFsbG9jYXRvcj4sIChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6RGVmYXVsdEFsbG9jYXRvcj46OnBhcnNlRW5jb2RpbmcoKaAG2wEoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OkFic3RyYWN0TWFuZ2xpbmdQYXJzZXI8KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpNYW5nbGluZ1BhcnNlcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+LCAoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+Ojpjb25zdW1lSWYoY2hhcimhBt0BKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpBYnN0cmFjdE1hbmdsaW5nUGFyc2VyPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6TWFuZ2xpbmdQYXJzZXI8KGFub255bW91cyBuYW1lc3BhY2UpOjpEZWZhdWx0QWxsb2NhdG9yPiwgKGFub255bW91cyBuYW1lc3BhY2UpOjpEZWZhdWx0QWxsb2NhdG9yPjo6cGFyc2VOdW1iZXIoYm9vbCmiBsEDKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpOb2RlKiAoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OkFic3RyYWN0TWFuZ2xpbmdQYXJzZXI8KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpNYW5nbGluZ1BhcnNlcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+LCAoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+OjptYWtlPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6U3BlY2lhbE5hbWUsIGNoYXIgY29uc3QgKCYpIFszNF0sIChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6Tm9kZSomPihjaGFyIGNvbnN0ICgmKSBbMzRdLCAoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok5vZGUqJimjBtcBKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpBYnN0cmFjdE1hbmdsaW5nUGFyc2VyPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6TWFuZ2xpbmdQYXJzZXI8KGFub255bW91cyBuYW1lc3BhY2UpOjpEZWZhdWx0QWxsb2NhdG9yPiwgKGFub255bW91cyBuYW1lc3BhY2UpOjpEZWZhdWx0QWxsb2NhdG9yPjo6cGFyc2VUeXBlKCmkBkooYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok91dHB1dEJ1ZmZlcjo6Z3Jvdyh1bnNpZ25lZCBsb25nKaUGkwEoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OlBPRFNtYWxsVmVjdG9yPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6Rm9yd2FyZFRlbXBsYXRlUmVmZXJlbmNlKiwgNHVsPjo6flBPRFNtYWxsVmVjdG9yKCmmBn8oYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OlBPRFNtYWxsVmVjdG9yPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6Tm9kZSosIDMydWw+OjpQT0RTbWFsbFZlY3RvcigppwZ+KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpQT0RTbWFsbFZlY3RvcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok5vZGUqLCA4dWw+OjpQT0RTbWFsbFZlY3RvcigpqAa9AShhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6UE9EU21hbGxWZWN0b3I8KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpQT0RTbWFsbFZlY3RvcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok5vZGUqLCA4dWw+KiwgNHVsPjo6UE9EU21hbGxWZWN0b3IoKakGeihhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6U3RyaW5nVmlldzo6c3RhcnRzV2l0aCgoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OlN0cmluZ1ZpZXcpIGNvbnN0qgatAyhhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6QWJzdHJhY3RNYW5nbGluZ1BhcnNlcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok1hbmdsaW5nUGFyc2VyPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6RGVmYXVsdEFsbG9jYXRvcj4sIChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6RGVmYXVsdEFsbG9jYXRvcj46OnBhcnNlTmFtZSgoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OkFic3RyYWN0TWFuZ2xpbmdQYXJzZXI8KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpNYW5nbGluZ1BhcnNlcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+LCAoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+OjpOYW1lU3RhdGUqKasG+wEoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OkFic3RyYWN0TWFuZ2xpbmdQYXJzZXI8KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpNYW5nbGluZ1BhcnNlcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+LCAoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+OjpwYXJzZUVuY29kaW5nKCk6OidsYW1iZGEnKCk6Om9wZXJhdG9yKCkoKSBjb25zdKwG3gEoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OkFic3RyYWN0TWFuZ2xpbmdQYXJzZXI8KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpNYW5nbGluZ1BhcnNlcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+LCAoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+OjpwYXJzZVRlbXBsYXRlQXJnKCmtBq8BKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpQT0RTbWFsbFZlY3RvcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok5vZGUqLCAzMnVsPjo6cHVzaF9iYWNrKChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6Tm9kZSogY29uc3QmKa4G7wEoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OkFic3RyYWN0TWFuZ2xpbmdQYXJzZXI8KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpNYW5nbGluZ1BhcnNlcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+LCAoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+Ojpwb3BUcmFpbGluZ05vZGVBcnJheSh1bnNpZ25lZCBsb25nKa8GrAcoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok5vZGUqIChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6QWJzdHJhY3RNYW5nbGluZ1BhcnNlcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok1hbmdsaW5nUGFyc2VyPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6RGVmYXVsdEFsbG9jYXRvcj4sIChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6RGVmYXVsdEFsbG9jYXRvcj46Om1ha2U8KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpGdW5jdGlvbkVuY29kaW5nLCAoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok5vZGUqJiwgKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpOb2RlKiYsIChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6Tm9kZUFycmF5LCAoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok5vZGUqJiwgKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpRdWFsaWZpZXJzJiwgKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpGdW5jdGlvblJlZlF1YWwmPigoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok5vZGUqJiwgKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpOb2RlKiYsIChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6Tm9kZUFycmF5JiYsIChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6Tm9kZSomLCAoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OlF1YWxpZmllcnMmLCAoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OkZ1bmN0aW9uUmVmUXVhbCYpsAaGAihhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6QWJzdHJhY3RNYW5nbGluZ1BhcnNlcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok1hbmdsaW5nUGFyc2VyPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6RGVmYXVsdEFsbG9jYXRvcj4sIChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6RGVmYXVsdEFsbG9jYXRvcj46OnBhcnNlRW5jb2RpbmcoKTo6U2F2ZVRlbXBsYXRlUGFyYW1zOjp+U2F2ZVRlbXBsYXRlUGFyYW1zKCmxBtoCKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpOb2RlKiAoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OkFic3RyYWN0TWFuZ2xpbmdQYXJzZXI8KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpNYW5nbGluZ1BhcnNlcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+LCAoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+OjptYWtlPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6TmFtZVR5cGUsIGNoYXIgY29uc3QgKCYpIFs1XT4oY2hhciBjb25zdCAoJikgWzVdKbIG4QEoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OkFic3RyYWN0TWFuZ2xpbmdQYXJzZXI8KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpNYW5nbGluZ1BhcnNlcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+LCAoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+OjpwYXJzZUJhcmVTb3VyY2VOYW1lKCmzBp4DKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpOb2RlKiAoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OkFic3RyYWN0TWFuZ2xpbmdQYXJzZXI8KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpNYW5nbGluZ1BhcnNlcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+LCAoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+OjptYWtlPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6TmFtZVR5cGUsIChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6U3RyaW5nVmlldyY+KChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6U3RyaW5nVmlldyYptAbXAShhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6QWJzdHJhY3RNYW5nbGluZ1BhcnNlcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok1hbmdsaW5nUGFyc2VyPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6RGVmYXVsdEFsbG9jYXRvcj4sIChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6RGVmYXVsdEFsbG9jYXRvcj46OnBhcnNlRXhwcigptQbbAShhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6QWJzdHJhY3RNYW5nbGluZ1BhcnNlcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok1hbmdsaW5nUGFyc2VyPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6RGVmYXVsdEFsbG9jYXRvcj4sIChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6RGVmYXVsdEFsbG9jYXRvcj46OnBhcnNlRGVjbHR5cGUoKbYGogMoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok5vZGUqIChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6QWJzdHJhY3RNYW5nbGluZ1BhcnNlcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok1hbmdsaW5nUGFyc2VyPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6RGVmYXVsdEFsbG9jYXRvcj4sIChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6RGVmYXVsdEFsbG9jYXRvcj46Om1ha2U8KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpQYXJhbWV0ZXJQYWNrRXhwYW5zaW9uLCAoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok5vZGUqJj4oKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpOb2RlKiYptwbgAShhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6QWJzdHJhY3RNYW5nbGluZ1BhcnNlcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok1hbmdsaW5nUGFyc2VyPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6RGVmYXVsdEFsbG9jYXRvcj4sIChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6RGVmYXVsdEFsbG9jYXRvcj46OnBhcnNlVGVtcGxhdGVQYXJhbSgpuAbjAShhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6QWJzdHJhY3RNYW5nbGluZ1BhcnNlcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok1hbmdsaW5nUGFyc2VyPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6RGVmYXVsdEFsbG9jYXRvcj4sIChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6RGVmYXVsdEFsbG9jYXRvcj46OnBhcnNlVGVtcGxhdGVBcmdzKGJvb2wpuQaCBChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6Tm9kZSogKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpBYnN0cmFjdE1hbmdsaW5nUGFyc2VyPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6TWFuZ2xpbmdQYXJzZXI8KGFub255bW91cyBuYW1lc3BhY2UpOjpEZWZhdWx0QWxsb2NhdG9yPiwgKGFub255bW91cyBuYW1lc3BhY2UpOjpEZWZhdWx0QWxsb2NhdG9yPjo6bWFrZTwoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok5hbWVXaXRoVGVtcGxhdGVBcmdzLCAoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok5vZGUqJiwgKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpOb2RlKiY+KChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6Tm9kZSomLCAoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok5vZGUqJim6BosEKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpOb2RlKiAoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OkFic3RyYWN0TWFuZ2xpbmdQYXJzZXI8KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpNYW5nbGluZ1BhcnNlcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+LCAoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+OjptYWtlPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6UmVmZXJlbmNlVHlwZSwgKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpOb2RlKiYsIChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6UmVmZXJlbmNlS2luZD4oKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpOb2RlKiYsIChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6UmVmZXJlbmNlS2luZCYmKbsGvAMoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OkFic3RyYWN0TWFuZ2xpbmdQYXJzZXI8KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpNYW5nbGluZ1BhcnNlcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+LCAoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+OjpwYXJzZVVuc2NvcGVkTmFtZSgoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OkFic3RyYWN0TWFuZ2xpbmdQYXJzZXI8KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpNYW5nbGluZ1BhcnNlcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+LCAoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+OjpOYW1lU3RhdGUqLCBib29sKim8BuABKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpBYnN0cmFjdE1hbmdsaW5nUGFyc2VyPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6TWFuZ2xpbmdQYXJzZXI8KGFub255bW91cyBuYW1lc3BhY2UpOjpEZWZhdWx0QWxsb2NhdG9yPiwgKGFub255bW91cyBuYW1lc3BhY2UpOjpEZWZhdWx0QWxsb2NhdG9yPjo6cGFyc2VRdWFsaWZpZWRUeXBlKCm9BuUCKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpQT0RTbWFsbFZlY3RvcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OlBPRFNtYWxsVmVjdG9yPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6Tm9kZSosIDh1bD4qLCA0dWw+OjpvcGVyYXRvcj0oKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpQT0RTbWFsbFZlY3RvcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OlBPRFNtYWxsVmVjdG9yPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6Tm9kZSosIDh1bD4qLCA0dWw+JiYpvgbnAShhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6UE9EU21hbGxWZWN0b3I8KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpOb2RlKiwgOHVsPjo6b3BlcmF0b3I9KChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6UE9EU21hbGxWZWN0b3I8KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpOb2RlKiwgOHVsPiYmKb8G3QEoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OkFic3RyYWN0TWFuZ2xpbmdQYXJzZXI8KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpNYW5nbGluZ1BhcnNlcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+LCAoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+OjpwYXJzZUNhbGxPZmZzZXQoKcAG5gEoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OkFic3RyYWN0TWFuZ2xpbmdQYXJzZXI8KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpNYW5nbGluZ1BhcnNlcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+LCAoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+OjpwYXJzZVNlcUlkKHVuc2lnbmVkIGxvbmcqKcEGlQIoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OkFic3RyYWN0TWFuZ2xpbmdQYXJzZXI8KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpNYW5nbGluZ1BhcnNlcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+LCAoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+OjpwYXJzZU1vZHVsZU5hbWVPcHQoKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpNb2R1bGVOYW1lKiYpwgabAShhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6UE9EU21hbGxWZWN0b3I8KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpGb3J3YXJkVGVtcGxhdGVSZWZlcmVuY2UqLCA0dWw+OjpvcGVyYXRvcltdKHVuc2lnbmVkIGxvbmcpwwaZAShhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6UE9EU21hbGxWZWN0b3I8KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpGb3J3YXJkVGVtcGxhdGVSZWZlcmVuY2UqLCA0dWw+Ojpkcm9wQmFjayh1bnNpZ25lZCBsb25nKcQG3gEoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OkFic3RyYWN0TWFuZ2xpbmdQYXJzZXI8KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpNYW5nbGluZ1BhcnNlcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+LCAoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+OjpwYXJzZUV4cHJQcmltYXJ5KCnFBrkFKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpQT0RTbWFsbFZlY3RvcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok5vZGUqLCA4dWw+Kiogc3RkOjpfXzI6OmNvcHlbYWJpOnYxNTAwN108KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpQT0RTbWFsbFZlY3RvcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok5vZGUqLCA4dWw+KiosIChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6UE9EU21hbGxWZWN0b3I8KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpOb2RlKiwgOHVsPioqPigoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OlBPRFNtYWxsVmVjdG9yPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6Tm9kZSosIDh1bD4qKiwgKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpQT0RTbWFsbFZlY3RvcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok5vZGUqLCA4dWw+KiosIChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6UE9EU21hbGxWZWN0b3I8KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpOb2RlKiwgOHVsPioqKcYGswMoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OkFic3RyYWN0TWFuZ2xpbmdQYXJzZXI8KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpNYW5nbGluZ1BhcnNlcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+LCAoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+OjpwYXJzZVNvdXJjZU5hbWUoKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpBYnN0cmFjdE1hbmdsaW5nUGFyc2VyPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6TWFuZ2xpbmdQYXJzZXI8KGFub255bW91cyBuYW1lc3BhY2UpOjpEZWZhdWx0QWxsb2NhdG9yPiwgKGFub255bW91cyBuYW1lc3BhY2UpOjpEZWZhdWx0QWxsb2NhdG9yPjo6TmFtZVN0YXRlKinHBkQoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkJ1bXBQb2ludGVyQWxsb2NhdG9yOjphbGxvY2F0ZSh1bnNpZ25lZCBsb25nKcgGrAEoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OlNwZWNpYWxOYW1lOjpTcGVjaWFsTmFtZSgoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OlN0cmluZ1ZpZXcsIChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6Tm9kZSBjb25zdCopyQa/Aihhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6Tm9kZTo6Tm9kZSgoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok5vZGU6OktpbmQsIChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6Tm9kZTo6UHJlYywgKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpOb2RlOjpDYWNoZSwgKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpOb2RlOjpDYWNoZSwgKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpOb2RlOjpDYWNoZSnKBn0oYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OlNwZWNpYWxOYW1lOjpwcmludExlZnQoKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpPdXRwdXRCdWZmZXImKSBjb25zdMsGdihhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6T3V0cHV0QnVmZmVyOjpvcGVyYXRvcis9KChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6U3RyaW5nVmlldynMBkIoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok5vZGU6OmdldEJhc2VOYW1lKCkgY29uc3TNBocBKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpDdG9yVnRhYmxlU3BlY2lhbE5hbWU6OnByaW50TGVmdCgoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok91dHB1dEJ1ZmZlciYpIGNvbnN0zgbwAShhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6QWJzdHJhY3RNYW5nbGluZ1BhcnNlcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok1hbmdsaW5nUGFyc2VyPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6RGVmYXVsdEFsbG9jYXRvcj4sIChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6RGVmYXVsdEFsbG9jYXRvcj46OnBhcnNlUG9zaXRpdmVJbnRlZ2VyKHVuc2lnbmVkIGxvbmcqKc8GcChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6TmFtZVR5cGU6Ok5hbWVUeXBlKChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6U3RyaW5nVmlldynQBnooYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok5hbWVUeXBlOjpwcmludExlZnQoKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpPdXRwdXRCdWZmZXImKSBjb25zdNEGRihhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6TmFtZVR5cGU6OmdldEJhc2VOYW1lKCkgY29uc3TSBnwoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok1vZHVsZU5hbWU6OnByaW50TGVmdCgoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok91dHB1dEJ1ZmZlciYpIGNvbnN00wbfAShhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6QWJzdHJhY3RNYW5nbGluZ1BhcnNlcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok1hbmdsaW5nUGFyc2VyPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6RGVmYXVsdEFsbG9jYXRvcj4sIChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6RGVmYXVsdEFsbG9jYXRvcj46OnBhcnNlQ1ZRdWFsaWZpZXJzKCnUBt8BKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpBYnN0cmFjdE1hbmdsaW5nUGFyc2VyPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6TWFuZ2xpbmdQYXJzZXI8KGFub255bW91cyBuYW1lc3BhY2UpOjpEZWZhdWx0QWxsb2NhdG9yPiwgKGFub255bW91cyBuYW1lc3BhY2UpOjpEZWZhdWx0QWxsb2NhdG9yPjo6cGFyc2VTdWJzdGl0dXRpb24oKdUGeShhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6UE9EU21hbGxWZWN0b3I8KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpOb2RlKiwgMzJ1bD46OnBvcF9iYWNrKCnWBp4EKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpBYnN0cmFjdE1hbmdsaW5nUGFyc2VyPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6TWFuZ2xpbmdQYXJzZXI8KGFub255bW91cyBuYW1lc3BhY2UpOjpEZWZhdWx0QWxsb2NhdG9yPiwgKGFub255bW91cyBuYW1lc3BhY2UpOjpEZWZhdWx0QWxsb2NhdG9yPjo6cGFyc2VVbnF1YWxpZmllZE5hbWUoKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpBYnN0cmFjdE1hbmdsaW5nUGFyc2VyPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6TWFuZ2xpbmdQYXJzZXI8KGFub255bW91cyBuYW1lc3BhY2UpOjpEZWZhdWx0QWxsb2NhdG9yPiwgKGFub255bW91cyBuYW1lc3BhY2UpOjpEZWZhdWx0QWxsb2NhdG9yPjo6TmFtZVN0YXRlKiwgKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpOb2RlKiwgKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpNb2R1bGVOYW1lKinXBlYoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OnBhcnNlX2Rpc2NyaW1pbmF0b3IoY2hhciBjb25zdCosIGNoYXIgY29uc3QqKdgG9wMoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok5vZGUqIChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6QWJzdHJhY3RNYW5nbGluZ1BhcnNlcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok1hbmdsaW5nUGFyc2VyPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6RGVmYXVsdEFsbG9jYXRvcj4sIChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6RGVmYXVsdEFsbG9jYXRvcj46Om1ha2U8KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpMb2NhbE5hbWUsIChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6Tm9kZSomLCAoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok5vZGUqJj4oKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpOb2RlKiYsIChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6Tm9kZSomKdkGswEoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OlBPRFNtYWxsVmVjdG9yPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6UE9EU21hbGxWZWN0b3I8KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpOb2RlKiwgOHVsPiosIDR1bD46OmJhY2soKdoGiAIoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OkFic3RyYWN0TWFuZ2xpbmdQYXJzZXI8KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpNYW5nbGluZ1BhcnNlcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+LCAoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+OjpwYXJzZUFiaVRhZ3MoKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpOb2RlKinbBrgDKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpBYnN0cmFjdE1hbmdsaW5nUGFyc2VyPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6TWFuZ2xpbmdQYXJzZXI8KGFub255bW91cyBuYW1lc3BhY2UpOjpEZWZhdWx0QWxsb2NhdG9yPiwgKGFub255bW91cyBuYW1lc3BhY2UpOjpEZWZhdWx0QWxsb2NhdG9yPjo6cGFyc2VVbm5hbWVkVHlwZU5hbWUoKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpBYnN0cmFjdE1hbmdsaW5nUGFyc2VyPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6TWFuZ2xpbmdQYXJzZXI8KGFub255bW91cyBuYW1lc3BhY2UpOjpEZWZhdWx0QWxsb2NhdG9yPiwgKGFub255bW91cyBuYW1lc3BhY2UpOjpEZWZhdWx0QWxsb2NhdG9yPjo6TmFtZVN0YXRlKincBrUDKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpBYnN0cmFjdE1hbmdsaW5nUGFyc2VyPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6TWFuZ2xpbmdQYXJzZXI8KGFub255bW91cyBuYW1lc3BhY2UpOjpEZWZhdWx0QWxsb2NhdG9yPiwgKGFub255bW91cyBuYW1lc3BhY2UpOjpEZWZhdWx0QWxsb2NhdG9yPjo6cGFyc2VPcGVyYXRvck5hbWUoKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpBYnN0cmFjdE1hbmdsaW5nUGFyc2VyPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6TWFuZ2xpbmdQYXJzZXI8KGFub255bW91cyBuYW1lc3BhY2UpOjpEZWZhdWx0QWxsb2NhdG9yPiwgKGFub255bW91cyBuYW1lc3BhY2UpOjpEZWZhdWx0QWxsb2NhdG9yPjo6TmFtZVN0YXRlKindBooCKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpOb2RlOjpOb2RlKChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6Tm9kZTo6S2luZCwgKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpOb2RlOjpDYWNoZSwgKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpOb2RlOjpDYWNoZSwgKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpOb2RlOjpDYWNoZSneBpQBKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpGb3J3YXJkVGVtcGxhdGVSZWZlcmVuY2U6Omhhc1JIU0NvbXBvbmVudFNsb3coKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpPdXRwdXRCdWZmZXImKSBjb25zdN8GfChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6Tm9kZTo6aGFzUkhTQ29tcG9uZW50KChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6T3V0cHV0QnVmZmVyJikgY29uc3TgBo0BKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpGb3J3YXJkVGVtcGxhdGVSZWZlcmVuY2U6Omhhc0FycmF5U2xvdygoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok91dHB1dEJ1ZmZlciYpIGNvbnN04QaQAShhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6Rm9yd2FyZFRlbXBsYXRlUmVmZXJlbmNlOjpoYXNGdW5jdGlvblNsb3coKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpPdXRwdXRCdWZmZXImKSBjb25zdOIGjgEoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OkZvcndhcmRUZW1wbGF0ZVJlZmVyZW5jZTo6Z2V0U3ludGF4Tm9kZSgoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok91dHB1dEJ1ZmZlciYpIGNvbnN04waKAShhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6Rm9yd2FyZFRlbXBsYXRlUmVmZXJlbmNlOjpwcmludExlZnQoKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpPdXRwdXRCdWZmZXImKSBjb25zdOQGiwEoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OkZvcndhcmRUZW1wbGF0ZVJlZmVyZW5jZTo6cHJpbnRSaWdodCgoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok91dHB1dEJ1ZmZlciYpIGNvbnN05QbjAShhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6QWJzdHJhY3RNYW5nbGluZ1BhcnNlcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok1hbmdsaW5nUGFyc2VyPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6RGVmYXVsdEFsbG9jYXRvcj4sIChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6RGVmYXVsdEFsbG9jYXRvcj46OnBhcnNlT3BlcmF0b3JFbmNvZGluZygp5gbrAShhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6QWJzdHJhY3RNYW5nbGluZ1BhcnNlcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok1hbmdsaW5nUGFyc2VyPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6RGVmYXVsdEFsbG9jYXRvcj4sIChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6RGVmYXVsdEFsbG9jYXRvcj46Ok9wZXJhdG9ySW5mbzo6Z2V0U3ltYm9sKCkgY29uc3TnBsUCKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpBYnN0cmFjdE1hbmdsaW5nUGFyc2VyPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6TWFuZ2xpbmdQYXJzZXI8KGFub255bW91cyBuYW1lc3BhY2UpOjpEZWZhdWx0QWxsb2NhdG9yPiwgKGFub255bW91cyBuYW1lc3BhY2UpOjpEZWZhdWx0QWxsb2NhdG9yPjo6cGFyc2VQcmVmaXhFeHByKChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6U3RyaW5nVmlldywgKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpOb2RlOjpQcmVjKegG6gQoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok5vZGUqIChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6QWJzdHJhY3RNYW5nbGluZ1BhcnNlcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok1hbmdsaW5nUGFyc2VyPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6RGVmYXVsdEFsbG9jYXRvcj4sIChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6RGVmYXVsdEFsbG9jYXRvcj46Om1ha2U8KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpDYWxsRXhwciwgKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpOb2RlKiYsIChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6Tm9kZUFycmF5LCAoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok5vZGU6OlByZWM+KChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6Tm9kZSomLCAoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok5vZGVBcnJheSYmLCAoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok5vZGU6OlByZWMmJinpBuABKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpBYnN0cmFjdE1hbmdsaW5nUGFyc2VyPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6TWFuZ2xpbmdQYXJzZXI8KGFub255bW91cyBuYW1lc3BhY2UpOjpEZWZhdWx0QWxsb2NhdG9yPiwgKGFub255bW91cyBuYW1lc3BhY2UpOjpEZWZhdWx0QWxsb2NhdG9yPjo6cGFyc2VGdW5jdGlvblBhcmFtKCnqBt0BKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpBYnN0cmFjdE1hbmdsaW5nUGFyc2VyPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6TWFuZ2xpbmdQYXJzZXI8KGFub255bW91cyBuYW1lc3BhY2UpOjpEZWZhdWx0QWxsb2NhdG9yPiwgKGFub255bW91cyBuYW1lc3BhY2UpOjpEZWZhdWx0QWxsb2NhdG9yPjo6cGFyc2VCcmFjZWRFeHByKCnrBsMDKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpOb2RlKiAoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OkFic3RyYWN0TWFuZ2xpbmdQYXJzZXI8KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpNYW5nbGluZ1BhcnNlcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+LCAoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+OjptYWtlPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6RW5jbG9zaW5nRXhwciwgY2hhciBjb25zdCAoJikgWzExXSwgKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpOb2RlKiY+KGNoYXIgY29uc3QgKCYpIFsxMV0sIChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6Tm9kZSomKewGqwEoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Om9wZXJhdG9yPT0oKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpTdHJpbmdWaWV3IGNvbnN0JiwgKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpTdHJpbmdWaWV3IGNvbnN0JintBlMoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OlN0cmluZ1ZpZXc6OmRyb3BGcm9udCh1bnNpZ25lZCBsb25nKSBjb25zdO4GlAIoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OkFic3RyYWN0TWFuZ2xpbmdQYXJzZXI8KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpNYW5nbGluZ1BhcnNlcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+LCAoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+OjpwYXJzZUludGVnZXJMaXRlcmFsKChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6U3RyaW5nVmlldynvBr4CKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpOb2RlKiAoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OkFic3RyYWN0TWFuZ2xpbmdQYXJzZXI8KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpNYW5nbGluZ1BhcnNlcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+LCAoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+OjptYWtlPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6Qm9vbEV4cHIsIGludD4oaW50JiYp8AajAyhhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6Tm9kZSogKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpBYnN0cmFjdE1hbmdsaW5nUGFyc2VyPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6TWFuZ2xpbmdQYXJzZXI8KGFub255bW91cyBuYW1lc3BhY2UpOjpEZWZhdWx0QWxsb2NhdG9yPiwgKGFub255bW91cyBuYW1lc3BhY2UpOjpEZWZhdWx0QWxsb2NhdG9yPjo6bWFrZTwoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OkZ1bmN0aW9uUGFyYW0sIChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6U3RyaW5nVmlldyY+KChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6U3RyaW5nVmlldyYp8QbpAShhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6QWJzdHJhY3RNYW5nbGluZ1BhcnNlcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok1hbmdsaW5nUGFyc2VyPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6RGVmYXVsdEFsbG9jYXRvcj4sIChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6RGVmYXVsdEFsbG9jYXRvcj46Ok9wZXJhdG9ySW5mbzo6Z2V0TmFtZSgpIGNvbnN08gaGBChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6Tm9kZSogKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpBYnN0cmFjdE1hbmdsaW5nUGFyc2VyPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6TWFuZ2xpbmdQYXJzZXI8KGFub255bW91cyBuYW1lc3BhY2UpOjpEZWZhdWx0QWxsb2NhdG9yPiwgKGFub255bW91cyBuYW1lc3BhY2UpOjpEZWZhdWx0QWxsb2NhdG9yPjo6bWFrZTwoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OkJyYWNlZEV4cHIsIChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6Tm9kZSomLCAoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok5vZGUqJiwgYm9vbD4oKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpOb2RlKiYsIChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6Tm9kZSomLCBib29sJiYp8wbhAShhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6QWJzdHJhY3RNYW5nbGluZ1BhcnNlcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok1hbmdsaW5nUGFyc2VyPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6RGVmYXVsdEFsbG9jYXRvcj4sIChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6RGVmYXVsdEFsbG9jYXRvcj46OnBhcnNlVW5yZXNvbHZlZFR5cGUoKfQG2wEoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OkFic3RyYWN0TWFuZ2xpbmdQYXJzZXI8KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpNYW5nbGluZ1BhcnNlcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+LCAoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+OjpwYXJzZVNpbXBsZUlkKCn1BvsDKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpOb2RlKiAoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OkFic3RyYWN0TWFuZ2xpbmdQYXJzZXI8KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpNYW5nbGluZ1BhcnNlcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+LCAoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+OjptYWtlPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6UXVhbGlmaWVkTmFtZSwgKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpOb2RlKiYsIChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6Tm9kZSomPigoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok5vZGUqJiwgKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpOb2RlKiYp9gblAShhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6QWJzdHJhY3RNYW5nbGluZ1BhcnNlcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok1hbmdsaW5nUGFyc2VyPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6RGVmYXVsdEFsbG9jYXRvcj4sIChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6RGVmYXVsdEFsbG9jYXRvcj46OnBhcnNlQmFzZVVucmVzb2x2ZWROYW1lKCn3Bp8DKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpOb2RlKiAoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OkFic3RyYWN0TWFuZ2xpbmdQYXJzZXI8KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpNYW5nbGluZ1BhcnNlcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+LCAoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+OjptYWtlPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6R2xvYmFsUXVhbGlmaWVkTmFtZSwgKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpOb2RlKiY+KChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6Tm9kZSomKfgGfChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6QmluYXJ5RXhwcjo6cHJpbnRMZWZ0KChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6T3V0cHV0QnVmZmVyJikgY29uc3T5BkYoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok91dHB1dEJ1ZmZlcjo6cHJpbnRPcGVuKGNoYXIp+ga2AShhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6Tm9kZTo6cHJpbnRBc09wZXJhbmQoKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpPdXRwdXRCdWZmZXImLCAoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok5vZGU6OlByZWMsIGJvb2wpIGNvbnN0+wZHKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpPdXRwdXRCdWZmZXI6OnByaW50Q2xvc2UoY2hhcin8BnwoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OlByZWZpeEV4cHI6OnByaW50TGVmdCgoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok91dHB1dEJ1ZmZlciYpIGNvbnN0/QZ9KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpQb3N0Zml4RXhwcjo6cHJpbnRMZWZ0KChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6T3V0cHV0QnVmZmVyJikgY29uc3T+BoQBKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpBcnJheVN1YnNjcmlwdEV4cHI6OnByaW50TGVmdCgoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok91dHB1dEJ1ZmZlciYpIGNvbnN0/wZ8KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpNZW1iZXJFeHByOjpwcmludExlZnQoKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpPdXRwdXRCdWZmZXImKSBjb25zdIAHeShhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6TmV3RXhwcjo6cHJpbnRMZWZ0KChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6T3V0cHV0QnVmZmVyJikgY29uc3SBB4ABKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpOb2RlQXJyYXk6OnByaW50V2l0aENvbW1hKChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6T3V0cHV0QnVmZmVyJikgY29uc3SCB3woYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OkRlbGV0ZUV4cHI6OnByaW50TGVmdCgoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok91dHB1dEJ1ZmZlciYpIGNvbnN0gwd6KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpDYWxsRXhwcjo6cHJpbnRMZWZ0KChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6T3V0cHV0QnVmZmVyJikgY29uc3SEB4ABKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpDb252ZXJzaW9uRXhwcjo6cHJpbnRMZWZ0KChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6T3V0cHV0QnVmZmVyJikgY29uc3SFB4EBKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpDb25kaXRpb25hbEV4cHI6OnByaW50TGVmdCgoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok91dHB1dEJ1ZmZlciYpIGNvbnN0hgd6KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpDYXN0RXhwcjo6cHJpbnRMZWZ0KChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6T3V0cHV0QnVmZmVyJikgY29uc3SHB+UBKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpFbmNsb3NpbmdFeHByOjpFbmNsb3NpbmdFeHByKChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6U3RyaW5nVmlldywgKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpOb2RlIGNvbnN0KiwgKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpOb2RlOjpQcmVjKYgHfyhhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6RW5jbG9zaW5nRXhwcjo6cHJpbnRMZWZ0KChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6T3V0cHV0QnVmZmVyJikgY29uc3SJB8kDKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpBYnN0cmFjdE1hbmdsaW5nUGFyc2VyPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6TWFuZ2xpbmdQYXJzZXI8KGFub255bW91cyBuYW1lc3BhY2UpOjpEZWZhdWx0QWxsb2NhdG9yPiwgKGFub255bW91cyBuYW1lc3BhY2UpOjpEZWZhdWx0QWxsb2NhdG9yPjo6U2NvcGVkVGVtcGxhdGVQYXJhbUxpc3Q6OlNjb3BlZFRlbXBsYXRlUGFyYW1MaXN0KChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6QWJzdHJhY3RNYW5nbGluZ1BhcnNlcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok1hbmdsaW5nUGFyc2VyPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6RGVmYXVsdEFsbG9jYXRvcj4sIChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6RGVmYXVsdEFsbG9jYXRvcj4qKYoH5AEoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OkFic3RyYWN0TWFuZ2xpbmdQYXJzZXI8KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpNYW5nbGluZ1BhcnNlcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+LCAoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+OjpwYXJzZVRlbXBsYXRlUGFyYW1EZWNsKCmLB/8BKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpBYnN0cmFjdE1hbmdsaW5nUGFyc2VyPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6TWFuZ2xpbmdQYXJzZXI8KGFub255bW91cyBuYW1lc3BhY2UpOjpEZWZhdWx0QWxsb2NhdG9yPiwgKGFub255bW91cyBuYW1lc3BhY2UpOjpEZWZhdWx0QWxsb2NhdG9yPjo6U2NvcGVkVGVtcGxhdGVQYXJhbUxpc3Q6On5TY29wZWRUZW1wbGF0ZVBhcmFtTGlzdCgpjAeAAShhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6SW50ZWdlckxpdGVyYWw6OnByaW50TGVmdCgoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok91dHB1dEJ1ZmZlciYpIGNvbnN0jQd6KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpCb29sRXhwcjo6cHJpbnRMZWZ0KChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6T3V0cHV0QnVmZmVyJikgY29uc3SOB4kBKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpGbG9hdExpdGVyYWxJbXBsPGZsb2F0Pjo6cHJpbnRMZWZ0KChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6T3V0cHV0QnVmZmVyJikgY29uc3SPB4oBKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpGbG9hdExpdGVyYWxJbXBsPGRvdWJsZT46OnByaW50TGVmdCgoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok91dHB1dEJ1ZmZlciYpIGNvbnN0kAePAShhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6RmxvYXRMaXRlcmFsSW1wbDxsb25nIGRvdWJsZT46OnByaW50TGVmdCgoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok91dHB1dEJ1ZmZlciYpIGNvbnN0kQd/KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpTdHJpbmdMaXRlcmFsOjpwcmludExlZnQoKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpPdXRwdXRCdWZmZXImKSBjb25zdJIH+AIoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OkFic3RyYWN0TWFuZ2xpbmdQYXJzZXI8KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpNYW5nbGluZ1BhcnNlcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+LCAoYW5vbnltb3VzIG5hbWVzcGFjZSk6OkRlZmF1bHRBbGxvY2F0b3I+OjpwYXJzZVRlbXBsYXRlUGFyYW1EZWNsKCk6OidsYW1iZGEnKChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6VGVtcGxhdGVQYXJhbUtpbmQpOjpvcGVyYXRvcigpKChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6VGVtcGxhdGVQYXJhbUtpbmQpIGNvbnN0kweBAShhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6VW5uYW1lZFR5cGVOYW1lOjpwcmludExlZnQoKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpPdXRwdXRCdWZmZXImKSBjb25zdJQHjAEoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OlN5bnRoZXRpY1RlbXBsYXRlUGFyYW1OYW1lOjpwcmludExlZnQoKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpPdXRwdXRCdWZmZXImKSBjb25zdJUHhwEoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OlR5cGVUZW1wbGF0ZVBhcmFtRGVjbDo6cHJpbnRMZWZ0KChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6T3V0cHV0QnVmZmVyJikgY29uc3SWB4gBKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpUeXBlVGVtcGxhdGVQYXJhbURlY2w6OnByaW50UmlnaHQoKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpPdXRwdXRCdWZmZXImKSBjb25zdJcHigEoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok5vblR5cGVUZW1wbGF0ZVBhcmFtRGVjbDo6cHJpbnRMZWZ0KChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6T3V0cHV0QnVmZmVyJikgY29uc3SYB4sBKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpOb25UeXBlVGVtcGxhdGVQYXJhbURlY2w6OnByaW50UmlnaHQoKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpPdXRwdXRCdWZmZXImKSBjb25zdJkHiwEoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OlRlbXBsYXRlVGVtcGxhdGVQYXJhbURlY2w6OnByaW50TGVmdCgoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok91dHB1dEJ1ZmZlciYpIGNvbnN0mgeHAShhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6VGVtcGxhdGVQYXJhbVBhY2tEZWNsOjpwcmludExlZnQoKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpPdXRwdXRCdWZmZXImKSBjb25zdJsHiAEoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OlRlbXBsYXRlUGFyYW1QYWNrRGVjbDo6cHJpbnRSaWdodCgoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok91dHB1dEJ1ZmZlciYpIGNvbnN0nAeBAShhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6Q2xvc3VyZVR5cGVOYW1lOjpwcmludExlZnQoKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpPdXRwdXRCdWZmZXImKSBjb25zdJ0HhwEoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OkNsb3N1cmVUeXBlTmFtZTo6cHJpbnREZWNsYXJhdG9yKChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6T3V0cHV0QnVmZmVyJikgY29uc3SeB3woYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OkxhbWJkYUV4cHI6OnByaW50TGVmdCgoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok91dHB1dEJ1ZmZlciYpIGNvbnN0nwd9KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpFbnVtTGl0ZXJhbDo6cHJpbnRMZWZ0KChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6T3V0cHV0QnVmZmVyJikgY29uc3SgB3YoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok91dHB1dEJ1ZmZlcjo6b3BlcmF0b3I8PCgoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OlN0cmluZ1ZpZXcpoQd/KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpGdW5jdGlvblBhcmFtOjpwcmludExlZnQoKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpPdXRwdXRCdWZmZXImKSBjb25zdKIHeihhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6Rm9sZEV4cHI6OnByaW50TGVmdCgoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok91dHB1dEJ1ZmZlciYpIGNvbnN0oweaAShhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6Rm9sZEV4cHI6OnByaW50TGVmdCgoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok91dHB1dEJ1ZmZlciYpIGNvbnN0OjonbGFtYmRhJygpOjpvcGVyYXRvcigpKCkgY29uc3SkB40BKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpQYXJhbWV0ZXJQYWNrRXhwYW5zaW9uOjpQYXJhbWV0ZXJQYWNrRXhwYW5zaW9uKChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6Tm9kZSBjb25zdCoppQeIAShhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6UGFyYW1ldGVyUGFja0V4cGFuc2lvbjo6cHJpbnRMZWZ0KChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6T3V0cHV0QnVmZmVyJikgY29uc3SmB3woYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OkJyYWNlZEV4cHI6OnByaW50TGVmdCgoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok91dHB1dEJ1ZmZlciYpIGNvbnN0pweBAShhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6QnJhY2VkUmFuZ2VFeHByOjpwcmludExlZnQoKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpPdXRwdXRCdWZmZXImKSBjb25zdKgHrQEoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OkluaXRMaXN0RXhwcjo6SW5pdExpc3RFeHByKChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6Tm9kZSBjb25zdCosIChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6Tm9kZUFycmF5KakHfihhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6SW5pdExpc3RFeHByOjpwcmludExlZnQoKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpPdXRwdXRCdWZmZXImKSBjb25zdKoHjwEoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OlBvaW50ZXJUb01lbWJlckNvbnZlcnNpb25FeHByOjpwcmludExlZnQoKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpPdXRwdXRCdWZmZXImKSBjb25zdKsHfyhhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6U3Vib2JqZWN0RXhwcjo6cHJpbnRMZWZ0KChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6T3V0cHV0QnVmZmVyJikgY29uc3SsB4UBKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpTaXplb2ZQYXJhbVBhY2tFeHByOjpwcmludExlZnQoKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpPdXRwdXRCdWZmZXImKSBjb25zdK0Hfyhhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6Tm9kZUFycmF5Tm9kZTo6cHJpbnRMZWZ0KChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6T3V0cHV0QnVmZmVyJikgY29uc3SuB3soYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OlRocm93RXhwcjo6cHJpbnRMZWZ0KChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6T3V0cHV0QnVmZmVyJikgY29uc3SvB38oYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OlF1YWxpZmllZE5hbWU6OnByaW50TGVmdCgoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok91dHB1dEJ1ZmZlciYpIGNvbnN0sAdLKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpRdWFsaWZpZWROYW1lOjpnZXRCYXNlTmFtZSgpIGNvbnN0sQeiAyhhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6Tm9kZSogKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpBYnN0cmFjdE1hbmdsaW5nUGFyc2VyPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6TWFuZ2xpbmdQYXJzZXI8KGFub255bW91cyBuYW1lc3BhY2UpOjpEZWZhdWx0QWxsb2NhdG9yPiwgKGFub255bW91cyBuYW1lc3BhY2UpOjpEZWZhdWx0QWxsb2NhdG9yPjo6bWFrZTwoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OkNvbnZlcnNpb25PcGVyYXRvclR5cGUsIChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6Tm9kZSomPigoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok5vZGUqJimyB3ooYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OkR0b3JOYW1lOjpwcmludExlZnQoKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpPdXRwdXRCdWZmZXImKSBjb25zdLMHiAEoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OkNvbnZlcnNpb25PcGVyYXRvclR5cGU6OnByaW50TGVmdCgoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok91dHB1dEJ1ZmZlciYpIGNvbnN0tAeBAShhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6TGl0ZXJhbE9wZXJhdG9yOjpwcmludExlZnQoKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpPdXRwdXRCdWZmZXImKSBjb25zdLUHhQEoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Okdsb2JhbFF1YWxpZmllZE5hbWU6OnByaW50TGVmdCgoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok91dHB1dEJ1ZmZlciYpIGNvbnN0tgdRKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpHbG9iYWxRdWFsaWZpZWROYW1lOjpnZXRCYXNlTmFtZSgpIGNvbnN0twfPAShhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6RXhwYW5kZWRTcGVjaWFsU3Vic3RpdHV0aW9uOjpFeHBhbmRlZFNwZWNpYWxTdWJzdGl0dXRpb24oKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpTcGVjaWFsU3ViS2luZCwgKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpOb2RlOjpLaW5kKbgHhQEoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OlNwZWNpYWxTdWJzdGl0dXRpb246OnByaW50TGVmdCgoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok91dHB1dEJ1ZmZlciYpIGNvbnN0uQdRKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpTcGVjaWFsU3Vic3RpdHV0aW9uOjpnZXRCYXNlTmFtZSgpIGNvbnN0ugdZKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpFeHBhbmRlZFNwZWNpYWxTdWJzdGl0dXRpb246OmdldEJhc2VOYW1lKCkgY29uc3S7B40BKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpFeHBhbmRlZFNwZWNpYWxTdWJzdGl0dXRpb246OnByaW50TGVmdCgoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok91dHB1dEJ1ZmZlciYpIGNvbnN0vAd8KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpBYmlUYWdBdHRyOjpwcmludExlZnQoKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpPdXRwdXRCdWZmZXImKSBjb25zdL0HsgMoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok5vZGUqIChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6QWJzdHJhY3RNYW5nbGluZ1BhcnNlcjwoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok1hbmdsaW5nUGFyc2VyPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6RGVmYXVsdEFsbG9jYXRvcj4sIChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6RGVmYXVsdEFsbG9jYXRvcj46Om1ha2U8KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpDdG9yRHRvck5hbWUsIChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6Tm9kZSomLCBib29sLCBpbnQmPigoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok5vZGUqJiwgYm9vbCYmLCBpbnQmKb4HhwEoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OlN0cnVjdHVyZWRCaW5kaW5nTmFtZTo6cHJpbnRMZWZ0KChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6T3V0cHV0QnVmZmVyJikgY29uc3S/B34oYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OkN0b3JEdG9yTmFtZTo6cHJpbnRMZWZ0KChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6T3V0cHV0QnVmZmVyJikgY29uc3TAB34oYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok1vZHVsZUVudGl0eTo6cHJpbnRMZWZ0KChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6T3V0cHV0QnVmZmVyJikgY29uc3TBB4kBKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpQYXJhbWV0ZXJQYWNrOjpoYXNSSFNDb21wb25lbnRTbG93KChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6T3V0cHV0QnVmZmVyJikgY29uc3TCB4IBKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpQYXJhbWV0ZXJQYWNrOjpoYXNBcnJheVNsb3coKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpPdXRwdXRCdWZmZXImKSBjb25zdMMHhQEoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OlBhcmFtZXRlclBhY2s6Omhhc0Z1bmN0aW9uU2xvdygoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok91dHB1dEJ1ZmZlciYpIGNvbnN0xAeDAShhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6UGFyYW1ldGVyUGFjazo6Z2V0U3ludGF4Tm9kZSgoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok91dHB1dEJ1ZmZlciYpIGNvbnN0xQd/KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpQYXJhbWV0ZXJQYWNrOjpwcmludExlZnQoKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpPdXRwdXRCdWZmZXImKSBjb25zdMYHgAEoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OlBhcmFtZXRlclBhY2s6OnByaW50UmlnaHQoKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpPdXRwdXRCdWZmZXImKSBjb25zdMcHfihhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6VGVtcGxhdGVBcmdzOjpwcmludExlZnQoKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpPdXRwdXRCdWZmZXImKSBjb25zdMgHhgEoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok5hbWVXaXRoVGVtcGxhdGVBcmdzOjpwcmludExlZnQoKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpPdXRwdXRCdWZmZXImKSBjb25zdMkHfihhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6RW5hYmxlSWZBdHRyOjpwcmludExlZnQoKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpPdXRwdXRCdWZmZXImKSBjb25zdMoHjAEoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OkZ1bmN0aW9uRW5jb2Rpbmc6Omhhc1JIU0NvbXBvbmVudFNsb3coKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpPdXRwdXRCdWZmZXImKSBjb25zdMsHggEoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OkZ1bmN0aW9uRW5jb2Rpbmc6OnByaW50TGVmdCgoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok91dHB1dEJ1ZmZlciYpIGNvbnN0zAeDAShhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6RnVuY3Rpb25FbmNvZGluZzo6cHJpbnRSaWdodCgoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok91dHB1dEJ1ZmZlciYpIGNvbnN0zQd7KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpEb3RTdWZmaXg6OnByaW50TGVmdCgoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok91dHB1dEJ1ZmZlciYpIGNvbnN0zgf4Ayhhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6Tm9kZSogKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpBYnN0cmFjdE1hbmdsaW5nUGFyc2VyPChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6TWFuZ2xpbmdQYXJzZXI8KGFub255bW91cyBuYW1lc3BhY2UpOjpEZWZhdWx0QWxsb2NhdG9yPiwgKGFub255bW91cyBuYW1lc3BhY2UpOjpEZWZhdWx0QWxsb2NhdG9yPjo6bWFrZTwoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OlZlY3RvclR5cGUsIChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6Tm9kZSomLCAoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok5vZGUqJj4oKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpOb2RlKiYsIChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6Tm9kZSomKc8Hfihhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6Tm9leGNlcHRTcGVjOjpwcmludExlZnQoKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpPdXRwdXRCdWZmZXImKSBjb25zdNAHhgEoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OkR5bmFtaWNFeGNlcHRpb25TcGVjOjpwcmludExlZnQoKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpPdXRwdXRCdWZmZXImKSBjb25zdNEHfihhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6RnVuY3Rpb25UeXBlOjpwcmludExlZnQoKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpPdXRwdXRCdWZmZXImKSBjb25zdNIHfyhhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6RnVuY3Rpb25UeXBlOjpwcmludFJpZ2h0KChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6T3V0cHV0QnVmZmVyJikgY29uc3TTB38oYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok9iakNQcm90b05hbWU6OnByaW50TGVmdCgoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok91dHB1dEJ1ZmZlciYpIGNvbnN01AeDAShhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6VmVuZG9yRXh0UXVhbFR5cGU6OnByaW50TGVmdCgoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok91dHB1dEJ1ZmZlciYpIGNvbnN01QeEAShhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6UXVhbFR5cGU6Omhhc1JIU0NvbXBvbmVudFNsb3coKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpPdXRwdXRCdWZmZXImKSBjb25zdNYHfShhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6UXVhbFR5cGU6Omhhc0FycmF5U2xvdygoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok91dHB1dEJ1ZmZlciYpIGNvbnN01weAAShhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6UXVhbFR5cGU6Omhhc0Z1bmN0aW9uU2xvdygoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok91dHB1dEJ1ZmZlciYpIGNvbnN02Ad6KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpRdWFsVHlwZTo6cHJpbnRMZWZ0KChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6T3V0cHV0QnVmZmVyJikgY29uc3TZB3soYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OlF1YWxUeXBlOjpwcmludFJpZ2h0KChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6T3V0cHV0QnVmZmVyJikgY29uc3TaB34oYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OkJpbmFyeUZQVHlwZTo6cHJpbnRMZWZ0KChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6T3V0cHV0QnVmZmVyJikgY29uc3TbB3woYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OkJpdEludFR5cGU6OnByaW50TGVmdCgoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok91dHB1dEJ1ZmZlciYpIGNvbnN03AeBAShhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6UGl4ZWxWZWN0b3JUeXBlOjpwcmludExlZnQoKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpPdXRwdXRCdWZmZXImKSBjb25zdN0HqwEoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OlZlY3RvclR5cGU6OlZlY3RvclR5cGUoKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpOb2RlIGNvbnN0KiwgKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpOb2RlIGNvbnN0KineB3woYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OlZlY3RvclR5cGU6OnByaW50TGVmdCgoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok91dHB1dEJ1ZmZlciYpIGNvbnN03wd7KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpBcnJheVR5cGU6OnByaW50TGVmdCgoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok91dHB1dEJ1ZmZlciYpIGNvbnN04Ad8KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpBcnJheVR5cGU6OnByaW50UmlnaHQoKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpPdXRwdXRCdWZmZXImKSBjb25zdOEHhQEoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OlBvaW50ZXJUb01lbWJlclR5cGU6OnByaW50TGVmdCgoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok91dHB1dEJ1ZmZlciYpIGNvbnN04geGAShhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6UG9pbnRlclRvTWVtYmVyVHlwZTo6cHJpbnRSaWdodCgoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok91dHB1dEJ1ZmZlciYpIGNvbnN04weIAShhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6RWxhYm9yYXRlZFR5cGVTcGVmVHlwZTo6cHJpbnRMZWZ0KChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6T3V0cHV0QnVmZmVyJikgY29uc3TkB4cBKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpQb2ludGVyVHlwZTo6aGFzUkhTQ29tcG9uZW50U2xvdygoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok91dHB1dEJ1ZmZlciYpIGNvbnN05Qd9KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpQb2ludGVyVHlwZTo6cHJpbnRMZWZ0KChhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6T3V0cHV0QnVmZmVyJikgY29uc3TmB0woYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok9iakNQcm90b05hbWU6OmlzT2JqQ09iamVjdCgpIGNvbnN05wd+KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpQb2ludGVyVHlwZTo6cHJpbnRSaWdodCgoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok91dHB1dEJ1ZmZlciYpIGNvbnN06Ad/KGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpSZWZlcmVuY2VUeXBlOjpwcmludExlZnQoKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpPdXRwdXRCdWZmZXImKSBjb25zdOkHfihhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6UmVmZXJlbmNlVHlwZTo6Y29sbGFwc2UoKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpPdXRwdXRCdWZmZXImKSBjb25zdOoHgAEoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OlJlZmVyZW5jZVR5cGU6OnByaW50UmlnaHQoKGFub255bW91cyBuYW1lc3BhY2UpOjppdGFuaXVtX2RlbWFuZ2xlOjpPdXRwdXRCdWZmZXImKSBjb25zdOsHvgEoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OlBvc3RmaXhRdWFsaWZpZWRUeXBlOjpQb3N0Zml4UXVhbGlmaWVkVHlwZSgoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok5vZGUgY29uc3QqLCAoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6OlN0cmluZ1ZpZXcp7AeGAShhbm9ueW1vdXMgbmFtZXNwYWNlKTo6aXRhbml1bV9kZW1hbmdsZTo6UG9zdGZpeFF1YWxpZmllZFR5cGU6OnByaW50TGVmdCgoYW5vbnltb3VzIG5hbWVzcGFjZSk6Oml0YW5pdW1fZGVtYW5nbGU6Ok91dHB1dEJ1ZmZlciYpIGNvbnN07QclX190aHJvd25fb2JqZWN0X2Zyb21fdW53aW5kX2V4Y2VwdGlvbu4HF19fZ2V0X2V4Y2VwdGlvbl9tZXNzYWdl7wcJc3RhY2tTYXZl8AcMc3RhY2tSZXN0b3Jl8QcKc3RhY2tBbGxvY/IHFmxlZ2Fsc3R1YiRkeW5DYWxsX2ppamnzBxhsZWdhbHN0dWIkZHluQ2FsbF92aWlqaWn0BxhsZWdhbHN0dWIkZHluQ2FsbF9paWlpaWr1BxlsZWdhbHN0dWIkZHluQ2FsbF9paWlpaWpq9gcabGVnYWxzdHViJGR5bkNhbGxfaWlpaWlpamr3ByFsZWdhbGZ1bmMkX2VtYmluZF9yZWdpc3Rlcl9iaWdpbnQHNwQAD19fc3RhY2tfcG9pbnRlcgELX19zdGFja19lbmQCDF9fc3RhY2tfYmFzZQMIdGVtcFJldDAJlwU7AAcucm9kYXRhAQkucm9kYXRhLjECCS5yb2RhdGEuMgMJLnJvZGF0YS4zBAkucm9kYXRhLjQFCS5yb2RhdGEuNQYJLnJvZGF0YS42Bwkucm9kYXRhLjcICS5yb2RhdGEuOAkJLnJvZGF0YS45Cgoucm9kYXRhLjEwCwoucm9kYXRhLjExDAoucm9kYXRhLjEyDQoucm9kYXRhLjEzDgoucm9kYXRhLjE0Dwoucm9kYXRhLjE1EAoucm9kYXRhLjE2EQoucm9kYXRhLjE3Egoucm9kYXRhLjE4Ewoucm9kYXRhLjE5FAoucm9kYXRhLjIwFQoucm9kYXRhLjIxFgoucm9kYXRhLjIyFwoucm9kYXRhLjIzGAoucm9kYXRhLjI0GQoucm9kYXRhLjI1Ggoucm9kYXRhLjI2Gwoucm9kYXRhLjI3HAoucm9kYXRhLjI4HQoucm9kYXRhLjI5Hgoucm9kYXRhLjMwHwoucm9kYXRhLjMxIAoucm9kYXRhLjMyIQoucm9kYXRhLjMzIgoucm9kYXRhLjM0Iwoucm9kYXRhLjM1JAoucm9kYXRhLjM2JQoucm9kYXRhLjM3Jgoucm9kYXRhLjM4Jwoucm9kYXRhLjM5KAoucm9kYXRhLjQwKQoucm9kYXRhLjQxKgoucm9kYXRhLjQyKwoucm9kYXRhLjQzLAoucm9kYXRhLjQ0LQoucm9kYXRhLjQ1Lgoucm9kYXRhLjQ2Lwoucm9kYXRhLjQ3MAUuZGF0YTEHLmRhdGEuMTIHLmRhdGEuMjMHLmRhdGEuMzQHLmRhdGEuNDUHLmRhdGEuNTYHLmRhdGEuNjcHLmRhdGEuNzgHLmRhdGEuODkHLmRhdGEuOToILmRhdGEuMTA=';
  if (!isDataURI(wasmBinaryFile)) {
    wasmBinaryFile = locateFile(wasmBinaryFile);
  }

function getBinary(file) {
  try {
    if (file == wasmBinaryFile && wasmBinary) {
      return new Uint8Array(wasmBinary);
    }
    var binary = tryParseAsDataURI(file);
    if (binary) {
      return binary;
    }
    if (readBinary) ;
    throw "both async and sync fetching of the wasm failed";
  }
  catch (err) {
    abort(err);
  }
}

function getBinaryPromise(binaryFile) {
  // If we don't have the binary yet, try to load it asynchronously.
  // Fetch has some additional restrictions over XHR, like it can't be used on a file:// url.
  // See https://github.com/github/fetch/pull/92#issuecomment-140665932
  // Cordova or Electron apps are typically loaded from a file:// url.
  // So use fetch if it is available and the url is not a file, otherwise fall back to XHR.
  if (!wasmBinary && (ENVIRONMENT_IS_WEB )) {
    if (typeof fetch == 'function'
    ) {
      return fetch(binaryFile, { credentials: 'same-origin' }).then(function(response) {
        if (!response['ok']) {
          throw "failed to load wasm binary file at '" + binaryFile + "'";
        }
        return response['arrayBuffer']();
      }).catch(function () {
          return getBinary(binaryFile);
      });
    }
  }

  // Otherwise, getBinary should be able to get it synchronously
  return Promise.resolve().then(function() { return getBinary(binaryFile); });
}

function instantiateArrayBuffer(binaryFile, imports, receiver) {
  return getBinaryPromise(binaryFile).then(function(binary) {
    return WebAssembly.instantiate(binary, imports);
  }).then(function (instance) {
    return instance;
  }).then(receiver, function(reason) {
    err('failed to asynchronously prepare wasm: ' + reason);

    // Warn on some common problems.
    if (isFileURI(wasmBinaryFile)) {
      err('warning: Loading from a file URI (' + wasmBinaryFile + ') is not supported in most browsers. See https://emscripten.org/docs/getting_started/FAQ.html#how-do-i-run-a-local-webserver-for-testing-why-does-my-program-stall-in-downloading-or-preparing');
    }
    abort(reason);
  });
}

function instantiateAsync(binary, binaryFile, imports, callback) {
  if (!binary &&
      typeof WebAssembly.instantiateStreaming == 'function' &&
      !isDataURI(binaryFile) &&
      typeof fetch == 'function') {
    return fetch(binaryFile, { credentials: 'same-origin' }).then(function(response) {
      // Suppress closure warning here since the upstream definition for
      // instantiateStreaming only allows Promise<Repsponse> rather than
      // an actual Response.
      // TODO(https://github.com/google/closure-compiler/pull/3913): Remove if/when upstream closure is fixed.
      /** @suppress {checkTypes} */
      var result = WebAssembly.instantiateStreaming(response, imports);

      return result.then(
        callback,
        function(reason) {
          // We expect the most common failure cause to be a bad MIME type for the binary,
          // in which case falling back to ArrayBuffer instantiation should work.
          err('wasm streaming compile failed: ' + reason);
          err('falling back to ArrayBuffer instantiation');
          return instantiateArrayBuffer(binaryFile, imports, callback);
        });
    });
  } else {
    return instantiateArrayBuffer(binaryFile, imports, callback);
  }
}

// Create the wasm instance.
// Receives the wasm imports, returns the exports.
function createWasm() {
  // prepare imports
  var info = {
    'env': wasmImports,
    'wasi_snapshot_preview1': wasmImports,
  };
  // Load the wasm module and create an instance of using native support in the JS engine.
  // handle a generated wasm instance, receiving its exports and
  // performing other necessary setup
  /** @param {WebAssembly.Module=} module*/
  function receiveInstance(instance, module) {
    var exports = instance.exports;

    Module['asm'] = exports;

    wasmMemory = Module['asm']['memory'];
    assert(wasmMemory, "memory not found in wasm exports");
    // This assertion doesn't hold when emscripten is run in --post-link
    // mode.
    // TODO(sbc): Read INITIAL_MEMORY out of the wasm file in post-link mode.
    //assert(wasmMemory.buffer.byteLength === 16777216);
    updateMemoryViews();

    wasmTable = Module['asm']['__indirect_function_table'];
    assert(wasmTable, "table not found in wasm exports");

    addOnInit(Module['asm']['__wasm_call_ctors']);

    removeRunDependency('wasm-instantiate');

    return exports;
  }
  // wait for the pthread pool (if any)
  addRunDependency('wasm-instantiate');

  // Prefer streaming instantiation if available.
  // Async compilation can be confusing when an error on the page overwrites Module
  // (for example, if the order of elements is wrong, and the one defining Module is
  // later), so we save Module and check it later.
  var trueModule = Module;
  function receiveInstantiationResult(result) {
    // 'result' is a ResultObject object which has both the module and instance.
    // receiveInstance() will swap in the exports (to Module.asm) so they can be called
    assert(Module === trueModule, 'the Module object should not be replaced during async compilation - perhaps the order of HTML elements is wrong?');
    trueModule = null;
    // TODO: Due to Closure regression https://github.com/google/closure-compiler/issues/3193, the above line no longer optimizes out down to the following line.
    // When the regression is fixed, can restore the above PTHREADS-enabled path.
    receiveInstance(result['instance']);
  }

  // User shell pages can write their own Module.instantiateWasm = function(imports, successCallback) callback
  // to manually instantiate the Wasm module themselves. This allows pages to
  // run the instantiation parallel to any other async startup actions they are
  // performing.
  // Also pthreads and wasm workers initialize the wasm instance through this
  // path.
  if (Module['instantiateWasm']) {

    try {
      return Module['instantiateWasm'](info, receiveInstance);
    } catch(e) {
      err('Module.instantiateWasm callback failed with error: ' + e);
        // If instantiation fails, reject the module ready promise.
        readyPromiseReject(e);
    }
  }

  // If instantiation fails, reject the module ready promise.
  instantiateAsync(wasmBinary, wasmBinaryFile, info, receiveInstantiationResult).catch(readyPromiseReject);
  return {}; // no exports yet; we'll fill them in later
}

// Globals used by JS i64 conversions (see makeSetValue)
var tempDouble;
var tempI64;

// include: runtime_debug.js
function legacyModuleProp(prop, newName) {
  if (!Object.getOwnPropertyDescriptor(Module, prop)) {
    Object.defineProperty(Module, prop, {
      configurable: true,
      get: function() {
        abort('Module.' + prop + ' has been replaced with plain ' + newName + ' (the initial value can be provided on Module, but after startup the value is only looked for on a local variable of that name)');
      }
    });
  }
}

function ignoredModuleProp(prop) {
  if (Object.getOwnPropertyDescriptor(Module, prop)) {
    abort('`Module.' + prop + '` was supplied but `' + prop + '` not included in INCOMING_MODULE_JS_API');
  }
}

// forcing the filesystem exports a few things by default
function isExportedByForceFilesystem(name) {
  return name === 'FS_createPath' ||
         name === 'FS_createDataFile' ||
         name === 'FS_createPreloadedFile' ||
         name === 'FS_unlink' ||
         name === 'addRunDependency' ||
         // The old FS has some functionality that WasmFS lacks.
         name === 'FS_createLazyFile' ||
         name === 'FS_createDevice' ||
         name === 'removeRunDependency';
}

function missingGlobal(sym, msg) {
  if (typeof globalThis !== 'undefined') {
    Object.defineProperty(globalThis, sym, {
      configurable: true,
      get: function() {
        warnOnce('`' + sym + '` is not longer defined by emscripten. ' + msg);
        return undefined;
      }
    });
  }
}

missingGlobal('buffer', 'Please use HEAP8.buffer or wasmMemory.buffer');

function missingLibrarySymbol(sym) {
  if (typeof globalThis !== 'undefined' && !Object.getOwnPropertyDescriptor(globalThis, sym)) {
    Object.defineProperty(globalThis, sym, {
      configurable: true,
      get: function() {
        // Can't `abort()` here because it would break code that does runtime
        // checks.  e.g. `if (typeof SDL === 'undefined')`.
        var msg = '`' + sym + '` is a library symbol and not included by default; add it to your library.js __deps or to DEFAULT_LIBRARY_FUNCS_TO_INCLUDE on the command line';
        // DEFAULT_LIBRARY_FUNCS_TO_INCLUDE requires the name as it appears in
        // library.js, which means $name for a JS name with no prefix, or name
        // for a JS name like _name.
        var librarySymbol = sym;
        if (!librarySymbol.startsWith('_')) {
          librarySymbol = '$' + sym;
        }
        msg += " (e.g. -sDEFAULT_LIBRARY_FUNCS_TO_INCLUDE=" + librarySymbol + ")";
        if (isExportedByForceFilesystem(sym)) {
          msg += '. Alternatively, forcing filesystem support (-sFORCE_FILESYSTEM) can export this for you';
        }
        warnOnce(msg);
        return undefined;
      }
    });
  }
  // Any symbol that is not included from the JS libary is also (by definition)
  // not exported on the Module object.
  unexportedRuntimeSymbol(sym);
}

function unexportedRuntimeSymbol(sym) {
  if (!Object.getOwnPropertyDescriptor(Module, sym)) {
    Object.defineProperty(Module, sym, {
      configurable: true,
      get: function() {
        var msg = "'" + sym + "' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)";
        if (isExportedByForceFilesystem(sym)) {
          msg += '. Alternatively, forcing filesystem support (-sFORCE_FILESYSTEM) can export this for you';
        }
        abort(msg);
      }
    });
  }
}

// end include: runtime_debug.js
// === Body ===

var ASM_CONSTS = {
  44983: ($0, $1) => { let type = UTF8ToString($0); let directory = UTF8ToString($1); let allocatedDir = _malloc(directory.length + 1); stringToUTF8(directory, allocatedDir, directory.length + 1); let jsAllocatedDir = UTF8ToString(allocatedDir); if (type == "IDBFS") { FS.mkdir(jsAllocatedDir); FS.mount(IDBFS, {}, jsAllocatedDir); console.log('EmscriptenFileSystemManager: Mounting IDBFS filesystem...\n'); } else { throw new Error('Unsupported filesystem type, IDBFS is supported: ' + type); } _free(allocatedDir); }
};
function syncIdb_js(populateFromFS) { try { FS.syncfs(populateFromFS, function(err) { setTimeout(function() { if (err) { console.error('b. jsFS Error: syncing FS:', err); Module.setIdbfsSynced(false); } else { console.log('b. jsFS synced successfully'); Module.setIdbfsSynced(true); } }, 1); }); } catch (err) { Module.setIdbfsSynced(false); } }

  function callRuntimeCallbacks(callbacks) {
      while (callbacks.length > 0) {
        // Pass the module as the first argument.
        callbacks.shift()(Module);
      }
    }

  
  function getCppExceptionTag() {
      // In static linking, tags are defined within the wasm module and are
      // exported, whereas in dynamic linking, tags are defined in library.js in
      // JS code and wasm modules import them.
      return Module['asm']['__cpp_exception'];
    }
  
  function getCppExceptionThrownObjectFromWebAssemblyException(ex) {
      // In Wasm EH, the value extracted from WebAssembly.Exception is a pointer
      // to the unwind header. Convert it to the actual thrown value.
      var unwind_header = ex.getArg(getCppExceptionTag(), 0);
      return ___thrown_object_from_unwind_exception(unwind_header);
    }

  function withStackSave(f) {
      var stack = stackSave();
      var ret = f();
      stackRestore(stack);
      return ret;
    }
  
  
  
  function lengthBytesUTF8(str) {
      var len = 0;
      for (var i = 0; i < str.length; ++i) {
        // Gotcha: charCodeAt returns a 16-bit word that is a UTF-16 encoded code
        // unit, not a Unicode code point of the character! So decode
        // UTF16->UTF32->UTF8.
        // See http://unicode.org/faq/utf_bom.html#utf16-3
        var c = str.charCodeAt(i); // possibly a lead surrogate
        if (c <= 0x7F) {
          len++;
        } else if (c <= 0x7FF) {
          len += 2;
        } else if (c >= 0xD800 && c <= 0xDFFF) {
          len += 4; ++i;
        } else {
          len += 3;
        }
      }
      return len;
    }
  
  function stringToUTF8Array(str, heap, outIdx, maxBytesToWrite) {
      // Parameter maxBytesToWrite is not optional. Negative values, 0, null,
      // undefined and false each don't write out any bytes.
      if (!(maxBytesToWrite > 0))
        return 0;
  
      var startIdx = outIdx;
      var endIdx = outIdx + maxBytesToWrite - 1; // -1 for string null terminator.
      for (var i = 0; i < str.length; ++i) {
        // Gotcha: charCodeAt returns a 16-bit word that is a UTF-16 encoded code
        // unit, not a Unicode code point of the character! So decode
        // UTF16->UTF32->UTF8.
        // See http://unicode.org/faq/utf_bom.html#utf16-3
        // For UTF8 byte structure, see http://en.wikipedia.org/wiki/UTF-8#Description
        // and https://www.ietf.org/rfc/rfc2279.txt
        // and https://tools.ietf.org/html/rfc3629
        var u = str.charCodeAt(i); // possibly a lead surrogate
        if (u >= 0xD800 && u <= 0xDFFF) {
          var u1 = str.charCodeAt(++i);
          u = 0x10000 + ((u & 0x3FF) << 10) | (u1 & 0x3FF);
        }
        if (u <= 0x7F) {
          if (outIdx >= endIdx) break;
          heap[outIdx++] = u;
        } else if (u <= 0x7FF) {
          if (outIdx + 1 >= endIdx) break;
          heap[outIdx++] = 0xC0 | (u >> 6);
          heap[outIdx++] = 0x80 | (u & 63);
        } else if (u <= 0xFFFF) {
          if (outIdx + 2 >= endIdx) break;
          heap[outIdx++] = 0xE0 | (u >> 12);
          heap[outIdx++] = 0x80 | ((u >> 6) & 63);
          heap[outIdx++] = 0x80 | (u & 63);
        } else {
          if (outIdx + 3 >= endIdx) break;
          if (u > 0x10FFFF) warnOnce('Invalid Unicode code point ' + ptrToString(u) + ' encountered when serializing a JS string to a UTF-8 string in wasm memory! (Valid unicode code points should be in range 0-0x10FFFF).');
          heap[outIdx++] = 0xF0 | (u >> 18);
          heap[outIdx++] = 0x80 | ((u >> 12) & 63);
          heap[outIdx++] = 0x80 | ((u >> 6) & 63);
          heap[outIdx++] = 0x80 | (u & 63);
        }
      }
      // Null-terminate the pointer to the buffer.
      heap[outIdx] = 0;
      return outIdx - startIdx;
    }
  function stringToUTF8(str, outPtr, maxBytesToWrite) {
      assert(typeof maxBytesToWrite == 'number', 'stringToUTF8(str, outPtr, maxBytesToWrite) is missing the third parameter that specifies the length of the output buffer!');
      return stringToUTF8Array(str, HEAPU8,outPtr, maxBytesToWrite);
    }
  function stringToUTF8OnStack(str) {
      var size = lengthBytesUTF8(str) + 1;
      var ret = stackAlloc(size);
      stringToUTF8(str, ret, size);
      return ret;
    }
  
  var UTF8Decoder = typeof TextDecoder != 'undefined' ? new TextDecoder('utf8') : undefined;
  
    /**
     * Given a pointer 'idx' to a null-terminated UTF8-encoded string in the given
     * array that contains uint8 values, returns a copy of that string as a
     * Javascript String object.
     * heapOrArray is either a regular array, or a JavaScript typed array view.
     * @param {number} idx
     * @param {number=} maxBytesToRead
     * @return {string}
     */
  function UTF8ArrayToString(heapOrArray, idx, maxBytesToRead) {
      var endIdx = idx + maxBytesToRead;
      var endPtr = idx;
      // TextDecoder needs to know the byte length in advance, it doesn't stop on
      // null terminator by itself.  Also, use the length info to avoid running tiny
      // strings through TextDecoder, since .subarray() allocates garbage.
      // (As a tiny code save trick, compare endPtr against endIdx using a negation,
      // so that undefined means Infinity)
      while (heapOrArray[endPtr] && !(endPtr >= endIdx)) ++endPtr;
  
      if (endPtr - idx > 16 && heapOrArray.buffer && UTF8Decoder) {
        return UTF8Decoder.decode(heapOrArray.subarray(idx, endPtr));
      }
      var str = '';
      // If building with TextDecoder, we have already computed the string length
      // above, so test loop end condition against that
      while (idx < endPtr) {
        // For UTF8 byte structure, see:
        // http://en.wikipedia.org/wiki/UTF-8#Description
        // https://www.ietf.org/rfc/rfc2279.txt
        // https://tools.ietf.org/html/rfc3629
        var u0 = heapOrArray[idx++];
        if (!(u0 & 0x80)) { str += String.fromCharCode(u0); continue; }
        var u1 = heapOrArray[idx++] & 63;
        if ((u0 & 0xE0) == 0xC0) { str += String.fromCharCode(((u0 & 31) << 6) | u1); continue; }
        var u2 = heapOrArray[idx++] & 63;
        if ((u0 & 0xF0) == 0xE0) {
          u0 = ((u0 & 15) << 12) | (u1 << 6) | u2;
        } else {
          if ((u0 & 0xF8) != 0xF0) warnOnce('Invalid UTF-8 leading byte ' + ptrToString(u0) + ' encountered when deserializing a UTF-8 string in wasm memory to a JS string!');
          u0 = ((u0 & 7) << 18) | (u1 << 12) | (u2 << 6) | (heapOrArray[idx++] & 63);
        }
  
        if (u0 < 0x10000) {
          str += String.fromCharCode(u0);
        } else {
          var ch = u0 - 0x10000;
          str += String.fromCharCode(0xD800 | (ch >> 10), 0xDC00 | (ch & 0x3FF));
        }
      }
      return str;
    }
  
  
    /**
     * Given a pointer 'ptr' to a null-terminated UTF8-encoded string in the
     * emscripten HEAP, returns a copy of that string as a Javascript String object.
     *
     * @param {number} ptr
     * @param {number=} maxBytesToRead - An optional length that specifies the
     *   maximum number of bytes to read. You can omit this parameter to scan the
     *   string until the first   byte. If maxBytesToRead is passed, and the string
     *   at [ptr, ptr+maxBytesToReadr[ contains a null byte in the middle, then the
     *   string will cut short at that byte index (i.e. maxBytesToRead will not
     *   produce a string of exact length [ptr, ptr+maxBytesToRead[) N.B. mixing
     *   frequent uses of UTF8ToString() with and without maxBytesToRead may throw
     *   JS JIT optimizations off, so it is worth to consider consistently using one
     * @return {string}
     */
  function UTF8ToString(ptr, maxBytesToRead) {
      assert(typeof ptr == 'number');
      return ptr ? UTF8ArrayToString(HEAPU8, ptr, maxBytesToRead) : '';
    }
  function demangle(func) {
      // If demangle has failed before, stop demangling any further function names
      // This avoids an infinite recursion with malloc()->abort()->stackTrace()->demangle()->malloc()->...
      demangle.recursionGuard = (demangle.recursionGuard|0)+1;
      if (demangle.recursionGuard > 1) return func;
      return withStackSave(function() {
        try {
          var s = func;
          if (s.startsWith('__Z'))
            s = s.substr(1);
          var buf = stringToUTF8OnStack(s);
          var status = stackAlloc(4);
          var ret = ___cxa_demangle(buf, 0, 0, status);
          if (HEAP32[((status)>>2)] === 0 && ret) {
            return UTF8ToString(ret);
          }
          // otherwise, libcxxabi failed
        } catch(e) {
        } finally {
          _free(ret);
          if (demangle.recursionGuard < 2) --demangle.recursionGuard;
        }
        // failure when using libcxxabi, don't demangle
        return func;
      });
    }

  
  
  
  
  function getExceptionMessageCommon(ptr) {
      return withStackSave(function() {
        var type_addr_addr = stackAlloc(4);
        var message_addr_addr = stackAlloc(4);
        ___get_exception_message(ptr, type_addr_addr, message_addr_addr);
        var type_addr = HEAPU32[((type_addr_addr)>>2)];
        var message_addr = HEAPU32[((message_addr_addr)>>2)];
        var type = UTF8ToString(type_addr);
        _free(type_addr);
        var message;
        if (message_addr) {
          message = UTF8ToString(message_addr);
          _free(message_addr);
        }
        return [type, message];
      });
    }
  function getExceptionMessage(ex) {
      var ptr = getCppExceptionThrownObjectFromWebAssemblyException(ex);
      return getExceptionMessageCommon(ptr);
    }
  Module["getExceptionMessage"] = getExceptionMessage;

  function intArrayToString(array) {
    var ret = [];
    for (var i = 0; i < array.length; i++) {
      var chr = array[i];
      if (chr > 0xFF) {
        assert(false, 'Character code ' + chr + ' (' + String.fromCharCode(chr) + ')  at offset ' + i + ' not in 0x00-0xFF.');
        chr &= 0xFF;
      }
      ret.push(String.fromCharCode(chr));
    }
    return ret.join('');
  }

  function ptrToString(ptr) {
      assert(typeof ptr === 'number');
      return '0x' + ptr.toString(16).padStart(8, '0');
    }
  
  function demangleAll(text) {
      var regex =
        /\b_Z[\w\d_]+/g;
      return text.replace(regex,
        function(x) {
          var y = demangle(x);
          return x === y ? x : (y + ' [' + x + ']');
        });
    }

  function warnOnce(text) {
      if (!warnOnce.shown) warnOnce.shown = {};
      if (!warnOnce.shown[text]) {
        warnOnce.shown[text] = 1;
        err(text);
      }
    }

  function ___assert_fail(condition, filename, line, func) {
      abort('Assertion failed: ' + UTF8ToString(condition) + ', at: ' + [filename ? UTF8ToString(filename) : 'unknown filename', line, func ? UTF8ToString(func) : 'unknown function']);
    }

  function setErrNo(value) {
      HEAP32[((___errno_location())>>2)] = value;
      return value;
    }
  
  var PATH = {isAbs:(path) => path.charAt(0) === '/',splitPath:(filename) => {
        var splitPathRe = /^(\/?|)([\s\S]*?)((?:\.{1,2}|[^\/]+?|)(\.[^.\/]*|))(?:[\/]*)$/;
        return splitPathRe.exec(filename).slice(1);
      },normalizeArray:(parts, allowAboveRoot) => {
        // if the path tries to go above the root, `up` ends up > 0
        var up = 0;
        for (var i = parts.length - 1; i >= 0; i--) {
          var last = parts[i];
          if (last === '.') {
            parts.splice(i, 1);
          } else if (last === '..') {
            parts.splice(i, 1);
            up++;
          } else if (up) {
            parts.splice(i, 1);
            up--;
          }
        }
        // if the path is allowed to go above the root, restore leading ..s
        if (allowAboveRoot) {
          for (; up; up--) {
            parts.unshift('..');
          }
        }
        return parts;
      },normalize:(path) => {
        var isAbsolute = PATH.isAbs(path),
            trailingSlash = path.substr(-1) === '/';
        // Normalize the path
        path = PATH.normalizeArray(path.split('/').filter((p) => !!p), !isAbsolute).join('/');
        if (!path && !isAbsolute) {
          path = '.';
        }
        if (path && trailingSlash) {
          path += '/';
        }
        return (isAbsolute ? '/' : '') + path;
      },dirname:(path) => {
        var result = PATH.splitPath(path),
            root = result[0],
            dir = result[1];
        if (!root && !dir) {
          // No dirname whatsoever
          return '.';
        }
        if (dir) {
          // It has a dirname, strip trailing slash
          dir = dir.substr(0, dir.length - 1);
        }
        return root + dir;
      },basename:(path) => {
        // EMSCRIPTEN return '/'' for '/', not an empty string
        if (path === '/') return '/';
        path = PATH.normalize(path);
        path = path.replace(/\/$/, "");
        var lastSlash = path.lastIndexOf('/');
        if (lastSlash === -1) return path;
        return path.substr(lastSlash+1);
      },join:function() {
        var paths = Array.prototype.slice.call(arguments);
        return PATH.normalize(paths.join('/'));
      },join2:(l, r) => {
        return PATH.normalize(l + '/' + r);
      }};
  
  function initRandomFill() {
      if (typeof crypto == 'object' && typeof crypto['getRandomValues'] == 'function') {
        // for modern web browsers
        return (view) => crypto.getRandomValues(view);
      } else
      // we couldn't find a proper implementation, as Math.random() is not suitable for /dev/random, see emscripten-core/emscripten/pull/7096
      abort("no cryptographic support found for randomDevice. consider polyfilling it if you want to use something insecure like Math.random(), e.g. put this in a --pre-js: var crypto = { getRandomValues: function(array) { for (var i = 0; i < array.length; i++) array[i] = (Math.random()*256)|0 } };");
    }
  function randomFill(view) {
      // Lazily init on the first invocation.
      return (randomFill = initRandomFill())(view);
    }
  
  
  
  var PATH_FS = {resolve:function() {
        var resolvedPath = '',
          resolvedAbsolute = false;
        for (var i = arguments.length - 1; i >= -1 && !resolvedAbsolute; i--) {
          var path = (i >= 0) ? arguments[i] : FS.cwd();
          // Skip empty and invalid entries
          if (typeof path != 'string') {
            throw new TypeError('Arguments to path.resolve must be strings');
          } else if (!path) {
            return ''; // an invalid portion invalidates the whole thing
          }
          resolvedPath = path + '/' + resolvedPath;
          resolvedAbsolute = PATH.isAbs(path);
        }
        // At this point the path should be resolved to a full absolute path, but
        // handle relative paths to be safe (might happen when process.cwd() fails)
        resolvedPath = PATH.normalizeArray(resolvedPath.split('/').filter((p) => !!p), !resolvedAbsolute).join('/');
        return ((resolvedAbsolute ? '/' : '') + resolvedPath) || '.';
      },relative:(from, to) => {
        from = PATH_FS.resolve(from).substr(1);
        to = PATH_FS.resolve(to).substr(1);
        function trim(arr) {
          var start = 0;
          for (; start < arr.length; start++) {
            if (arr[start] !== '') break;
          }
          var end = arr.length - 1;
          for (; end >= 0; end--) {
            if (arr[end] !== '') break;
          }
          if (start > end) return [];
          return arr.slice(start, end - start + 1);
        }
        var fromParts = trim(from.split('/'));
        var toParts = trim(to.split('/'));
        var length = Math.min(fromParts.length, toParts.length);
        var samePartsLength = length;
        for (var i = 0; i < length; i++) {
          if (fromParts[i] !== toParts[i]) {
            samePartsLength = i;
            break;
          }
        }
        var outputParts = [];
        for (var i = samePartsLength; i < fromParts.length; i++) {
          outputParts.push('..');
        }
        outputParts = outputParts.concat(toParts.slice(samePartsLength));
        return outputParts.join('/');
      }};
  
  
  
  /** @type {function(string, boolean=, number=)} */
  function intArrayFromString(stringy, dontAddNull, length) {
    var len = length > 0 ? length : lengthBytesUTF8(stringy)+1;
    var u8array = new Array(len);
    var numBytesWritten = stringToUTF8Array(stringy, u8array, 0, u8array.length);
    if (dontAddNull) u8array.length = numBytesWritten;
    return u8array;
  }
  var TTY = {ttys:[],init:function () {
        // https://github.com/emscripten-core/emscripten/pull/1555
        // if (ENVIRONMENT_IS_NODE) {
        //   // currently, FS.init does not distinguish if process.stdin is a file or TTY
        //   // device, it always assumes it's a TTY device. because of this, we're forcing
        //   // process.stdin to UTF8 encoding to at least make stdin reading compatible
        //   // with text files until FS.init can be refactored.
        //   process.stdin.setEncoding('utf8');
        // }
      },shutdown:function() {
        // https://github.com/emscripten-core/emscripten/pull/1555
        // if (ENVIRONMENT_IS_NODE) {
        //   // inolen: any idea as to why node -e 'process.stdin.read()' wouldn't exit immediately (with process.stdin being a tty)?
        //   // isaacs: because now it's reading from the stream, you've expressed interest in it, so that read() kicks off a _read() which creates a ReadReq operation
        //   // inolen: I thought read() in that case was a synchronous operation that just grabbed some amount of buffered data if it exists?
        //   // isaacs: it is. but it also triggers a _read() call, which calls readStart() on the handle
        //   // isaacs: do process.stdin.pause() and i'd think it'd probably close the pending call
        //   process.stdin.pause();
        // }
      },register:function(dev, ops) {
        TTY.ttys[dev] = { input: [], output: [], ops: ops };
        FS.registerDevice(dev, TTY.stream_ops);
      },stream_ops:{open:function(stream) {
          var tty = TTY.ttys[stream.node.rdev];
          if (!tty) {
            throw new FS.ErrnoError(43);
          }
          stream.tty = tty;
          stream.seekable = false;
        },close:function(stream) {
          // flush any pending line data
          stream.tty.ops.fsync(stream.tty);
        },fsync:function(stream) {
          stream.tty.ops.fsync(stream.tty);
        },read:function(stream, buffer, offset, length, pos /* ignored */) {
          if (!stream.tty || !stream.tty.ops.get_char) {
            throw new FS.ErrnoError(60);
          }
          var bytesRead = 0;
          for (var i = 0; i < length; i++) {
            var result;
            try {
              result = stream.tty.ops.get_char(stream.tty);
            } catch (e) {
              throw new FS.ErrnoError(29);
            }
            if (result === undefined && bytesRead === 0) {
              throw new FS.ErrnoError(6);
            }
            if (result === null || result === undefined) break;
            bytesRead++;
            buffer[offset+i] = result;
          }
          if (bytesRead) {
            stream.node.timestamp = Date.now();
          }
          return bytesRead;
        },write:function(stream, buffer, offset, length, pos) {
          if (!stream.tty || !stream.tty.ops.put_char) {
            throw new FS.ErrnoError(60);
          }
          try {
            for (var i = 0; i < length; i++) {
              stream.tty.ops.put_char(stream.tty, buffer[offset+i]);
            }
          } catch (e) {
            throw new FS.ErrnoError(29);
          }
          if (length) {
            stream.node.timestamp = Date.now();
          }
          return i;
        }},default_tty_ops:{get_char:function(tty) {
          if (!tty.input.length) {
            var result = null;
            if (typeof window != 'undefined' &&
              typeof window.prompt == 'function') {
              // Browser.
              result = window.prompt('Input: ');  // returns null on cancel
              if (result !== null) {
                result += '\n';
              }
            } else if (typeof readline == 'function') {
              // Command line.
              result = readline();
              if (result !== null) {
                result += '\n';
              }
            }
            if (!result) {
              return null;
            }
            tty.input = intArrayFromString(result, true);
          }
          return tty.input.shift();
        },put_char:function(tty, val) {
          if (val === null || val === 10) {
            out(UTF8ArrayToString(tty.output, 0));
            tty.output = [];
          } else {
            if (val != 0) tty.output.push(val); // val == 0 would cut text output off in the middle.
          }
        },fsync:function(tty) {
          if (tty.output && tty.output.length > 0) {
            out(UTF8ArrayToString(tty.output, 0));
            tty.output = [];
          }
        }},default_tty1_ops:{put_char:function(tty, val) {
          if (val === null || val === 10) {
            err(UTF8ArrayToString(tty.output, 0));
            tty.output = [];
          } else {
            if (val != 0) tty.output.push(val);
          }
        },fsync:function(tty) {
          if (tty.output && tty.output.length > 0) {
            err(UTF8ArrayToString(tty.output, 0));
            tty.output = [];
          }
        }}};
  function mmapAlloc(size) {
      abort('internal error: mmapAlloc called but `emscripten_builtin_memalign` native symbol not exported');
    }
  var MEMFS = {ops_table:null,mount:function(mount) {
        return MEMFS.createNode(null, '/', 16384 | 511 /* 0777 */, 0);
      },createNode:function(parent, name, mode, dev) {
        if (FS.isBlkdev(mode) || FS.isFIFO(mode)) {
          // no supported
          throw new FS.ErrnoError(63);
        }
        if (!MEMFS.ops_table) {
          MEMFS.ops_table = {
            dir: {
              node: {
                getattr: MEMFS.node_ops.getattr,
                setattr: MEMFS.node_ops.setattr,
                lookup: MEMFS.node_ops.lookup,
                mknod: MEMFS.node_ops.mknod,
                rename: MEMFS.node_ops.rename,
                unlink: MEMFS.node_ops.unlink,
                rmdir: MEMFS.node_ops.rmdir,
                readdir: MEMFS.node_ops.readdir,
                symlink: MEMFS.node_ops.symlink
              },
              stream: {
                llseek: MEMFS.stream_ops.llseek
              }
            },
            file: {
              node: {
                getattr: MEMFS.node_ops.getattr,
                setattr: MEMFS.node_ops.setattr
              },
              stream: {
                llseek: MEMFS.stream_ops.llseek,
                read: MEMFS.stream_ops.read,
                write: MEMFS.stream_ops.write,
                allocate: MEMFS.stream_ops.allocate,
                mmap: MEMFS.stream_ops.mmap,
                msync: MEMFS.stream_ops.msync
              }
            },
            link: {
              node: {
                getattr: MEMFS.node_ops.getattr,
                setattr: MEMFS.node_ops.setattr,
                readlink: MEMFS.node_ops.readlink
              },
              stream: {}
            },
            chrdev: {
              node: {
                getattr: MEMFS.node_ops.getattr,
                setattr: MEMFS.node_ops.setattr
              },
              stream: FS.chrdev_stream_ops
            }
          };
        }
        var node = FS.createNode(parent, name, mode, dev);
        if (FS.isDir(node.mode)) {
          node.node_ops = MEMFS.ops_table.dir.node;
          node.stream_ops = MEMFS.ops_table.dir.stream;
          node.contents = {};
        } else if (FS.isFile(node.mode)) {
          node.node_ops = MEMFS.ops_table.file.node;
          node.stream_ops = MEMFS.ops_table.file.stream;
          node.usedBytes = 0; // The actual number of bytes used in the typed array, as opposed to contents.length which gives the whole capacity.
          // When the byte data of the file is populated, this will point to either a typed array, or a normal JS array. Typed arrays are preferred
          // for performance, and used by default. However, typed arrays are not resizable like normal JS arrays are, so there is a small disk size
          // penalty involved for appending file writes that continuously grow a file similar to std::vector capacity vs used -scheme.
          node.contents = null; 
        } else if (FS.isLink(node.mode)) {
          node.node_ops = MEMFS.ops_table.link.node;
          node.stream_ops = MEMFS.ops_table.link.stream;
        } else if (FS.isChrdev(node.mode)) {
          node.node_ops = MEMFS.ops_table.chrdev.node;
          node.stream_ops = MEMFS.ops_table.chrdev.stream;
        }
        node.timestamp = Date.now();
        // add the new node to the parent
        if (parent) {
          parent.contents[name] = node;
          parent.timestamp = node.timestamp;
        }
        return node;
      },getFileDataAsTypedArray:function(node) {
        if (!node.contents) return new Uint8Array(0);
        if (node.contents.subarray) return node.contents.subarray(0, node.usedBytes); // Make sure to not return excess unused bytes.
        return new Uint8Array(node.contents);
      },expandFileStorage:function(node, newCapacity) {
        var prevCapacity = node.contents ? node.contents.length : 0;
        if (prevCapacity >= newCapacity) return; // No need to expand, the storage was already large enough.
        // Don't expand strictly to the given requested limit if it's only a very small increase, but instead geometrically grow capacity.
        // For small filesizes (<1MB), perform size*2 geometric increase, but for large sizes, do a much more conservative size*1.125 increase to
        // avoid overshooting the allocation cap by a very large margin.
        var CAPACITY_DOUBLING_MAX = 1024 * 1024;
        newCapacity = Math.max(newCapacity, (prevCapacity * (prevCapacity < CAPACITY_DOUBLING_MAX ? 2.0 : 1.125)) >>> 0);
        if (prevCapacity != 0) newCapacity = Math.max(newCapacity, 256); // At minimum allocate 256b for each file when expanding.
        var oldContents = node.contents;
        node.contents = new Uint8Array(newCapacity); // Allocate new storage.
        if (node.usedBytes > 0) node.contents.set(oldContents.subarray(0, node.usedBytes), 0); // Copy old data over to the new storage.
      },resizeFileStorage:function(node, newSize) {
        if (node.usedBytes == newSize) return;
        if (newSize == 0) {
          node.contents = null; // Fully decommit when requesting a resize to zero.
          node.usedBytes = 0;
        } else {
          var oldContents = node.contents;
          node.contents = new Uint8Array(newSize); // Allocate new storage.
          if (oldContents) {
            node.contents.set(oldContents.subarray(0, Math.min(newSize, node.usedBytes))); // Copy old data over to the new storage.
          }
          node.usedBytes = newSize;
        }
      },node_ops:{getattr:function(node) {
          var attr = {};
          // device numbers reuse inode numbers.
          attr.dev = FS.isChrdev(node.mode) ? node.id : 1;
          attr.ino = node.id;
          attr.mode = node.mode;
          attr.nlink = 1;
          attr.uid = 0;
          attr.gid = 0;
          attr.rdev = node.rdev;
          if (FS.isDir(node.mode)) {
            attr.size = 4096;
          } else if (FS.isFile(node.mode)) {
            attr.size = node.usedBytes;
          } else if (FS.isLink(node.mode)) {
            attr.size = node.link.length;
          } else {
            attr.size = 0;
          }
          attr.atime = new Date(node.timestamp);
          attr.mtime = new Date(node.timestamp);
          attr.ctime = new Date(node.timestamp);
          // NOTE: In our implementation, st_blocks = Math.ceil(st_size/st_blksize),
          //       but this is not required by the standard.
          attr.blksize = 4096;
          attr.blocks = Math.ceil(attr.size / attr.blksize);
          return attr;
        },setattr:function(node, attr) {
          if (attr.mode !== undefined) {
            node.mode = attr.mode;
          }
          if (attr.timestamp !== undefined) {
            node.timestamp = attr.timestamp;
          }
          if (attr.size !== undefined) {
            MEMFS.resizeFileStorage(node, attr.size);
          }
        },lookup:function(parent, name) {
          throw FS.genericErrors[44];
        },mknod:function(parent, name, mode, dev) {
          return MEMFS.createNode(parent, name, mode, dev);
        },rename:function(old_node, new_dir, new_name) {
          // if we're overwriting a directory at new_name, make sure it's empty.
          if (FS.isDir(old_node.mode)) {
            var new_node;
            try {
              new_node = FS.lookupNode(new_dir, new_name);
            } catch (e) {
            }
            if (new_node) {
              for (var i in new_node.contents) {
                throw new FS.ErrnoError(55);
              }
            }
          }
          // do the internal rewiring
          delete old_node.parent.contents[old_node.name];
          old_node.parent.timestamp = Date.now();
          old_node.name = new_name;
          new_dir.contents[new_name] = old_node;
          new_dir.timestamp = old_node.parent.timestamp;
          old_node.parent = new_dir;
        },unlink:function(parent, name) {
          delete parent.contents[name];
          parent.timestamp = Date.now();
        },rmdir:function(parent, name) {
          var node = FS.lookupNode(parent, name);
          for (var i in node.contents) {
            throw new FS.ErrnoError(55);
          }
          delete parent.contents[name];
          parent.timestamp = Date.now();
        },readdir:function(node) {
          var entries = ['.', '..'];
          for (var key in node.contents) {
            if (!node.contents.hasOwnProperty(key)) {
              continue;
            }
            entries.push(key);
          }
          return entries;
        },symlink:function(parent, newname, oldpath) {
          var node = MEMFS.createNode(parent, newname, 511 /* 0777 */ | 40960, 0);
          node.link = oldpath;
          return node;
        },readlink:function(node) {
          if (!FS.isLink(node.mode)) {
            throw new FS.ErrnoError(28);
          }
          return node.link;
        }},stream_ops:{read:function(stream, buffer, offset, length, position) {
          var contents = stream.node.contents;
          if (position >= stream.node.usedBytes) return 0;
          var size = Math.min(stream.node.usedBytes - position, length);
          assert(size >= 0);
          if (size > 8 && contents.subarray) { // non-trivial, and typed array
            buffer.set(contents.subarray(position, position + size), offset);
          } else {
            for (var i = 0; i < size; i++) buffer[offset + i] = contents[position + i];
          }
          return size;
        },write:function(stream, buffer, offset, length, position, canOwn) {
          // The data buffer should be a typed array view
          assert(!(buffer instanceof ArrayBuffer));
          // If the buffer is located in main memory (HEAP), and if
          // memory can grow, we can't hold on to references of the
          // memory buffer, as they may get invalidated. That means we
          // need to do copy its contents.
          if (buffer.buffer === HEAP8.buffer) {
            canOwn = false;
          }
  
          if (!length) return 0;
          var node = stream.node;
          node.timestamp = Date.now();
  
          if (buffer.subarray && (!node.contents || node.contents.subarray)) { // This write is from a typed array to a typed array?
            if (canOwn) {
              assert(position === 0, 'canOwn must imply no weird position inside the file');
              node.contents = buffer.subarray(offset, offset + length);
              node.usedBytes = length;
              return length;
            } else if (node.usedBytes === 0 && position === 0) { // If this is a simple first write to an empty file, do a fast set since we don't need to care about old data.
              node.contents = buffer.slice(offset, offset + length);
              node.usedBytes = length;
              return length;
            } else if (position + length <= node.usedBytes) { // Writing to an already allocated and used subrange of the file?
              node.contents.set(buffer.subarray(offset, offset + length), position);
              return length;
            }
          }
  
          // Appending to an existing file and we need to reallocate, or source data did not come as a typed array.
          MEMFS.expandFileStorage(node, position+length);
          if (node.contents.subarray && buffer.subarray) {
            // Use typed array write which is available.
            node.contents.set(buffer.subarray(offset, offset + length), position);
          } else {
            for (var i = 0; i < length; i++) {
             node.contents[position + i] = buffer[offset + i]; // Or fall back to manual write if not.
            }
          }
          node.usedBytes = Math.max(node.usedBytes, position + length);
          return length;
        },llseek:function(stream, offset, whence) {
          var position = offset;
          if (whence === 1) {
            position += stream.position;
          } else if (whence === 2) {
            if (FS.isFile(stream.node.mode)) {
              position += stream.node.usedBytes;
            }
          }
          if (position < 0) {
            throw new FS.ErrnoError(28);
          }
          return position;
        },allocate:function(stream, offset, length) {
          MEMFS.expandFileStorage(stream.node, offset + length);
          stream.node.usedBytes = Math.max(stream.node.usedBytes, offset + length);
        },mmap:function(stream, length, position, prot, flags) {
          if (!FS.isFile(stream.node.mode)) {
            throw new FS.ErrnoError(43);
          }
          var ptr;
          var allocated;
          var contents = stream.node.contents;
          // Only make a new copy when MAP_PRIVATE is specified.
          if (!(flags & 2) && contents.buffer === HEAP8.buffer) {
            // We can't emulate MAP_SHARED when the file is not backed by the
            // buffer we're mapping to (e.g. the HEAP buffer).
            allocated = false;
            ptr = contents.byteOffset;
          } else {
            // Try to avoid unnecessary slices.
            if (position > 0 || position + length < contents.length) {
              if (contents.subarray) {
                contents = contents.subarray(position, position + length);
              } else {
                contents = Array.prototype.slice.call(contents, position, position + length);
              }
            }
            allocated = true;
            ptr = mmapAlloc();
            if (!ptr) {
              throw new FS.ErrnoError(48);
            }
            HEAP8.set(contents, ptr);
          }
          return { ptr: ptr, allocated: allocated };
        },msync:function(stream, buffer, offset, length, mmapFlags) {
          MEMFS.stream_ops.write(stream, buffer, 0, length, offset, false);
          // should we check if bytesWritten and length are the same?
          return 0;
        }}};
  
  /** @param {boolean=} noRunDep */
  function asyncLoad(url, onload, onerror, noRunDep) {
      var dep = !noRunDep ? getUniqueRunDependency('al ' + url) : '';
      readAsync(url, (arrayBuffer) => {
        assert(arrayBuffer, 'Loading data file "' + url + '" failed (no arrayBuffer).');
        onload(new Uint8Array(arrayBuffer));
        if (dep) removeRunDependency(dep);
      }, (event) => {
        if (onerror) {
          onerror();
        } else {
          throw 'Loading data file "' + url + '" failed.';
        }
      });
      if (dep) addRunDependency(dep);
    }
  
  
  
  
  
  
  var IDBFS = {dbs:{},indexedDB:() => {
        if (typeof indexedDB != 'undefined') return indexedDB;
        var ret = null;
        if (typeof window == 'object') ret = window.indexedDB || window.mozIndexedDB || window.webkitIndexedDB || window.msIndexedDB;
        assert(ret, 'IDBFS used, but indexedDB not supported');
        return ret;
      },DB_VERSION:21,DB_STORE_NAME:"FILE_DATA",mount:function(mount) {
        // reuse all of the core MEMFS functionality
        return MEMFS.mount.apply(null, arguments);
      },syncfs:(mount, populate, callback) => {
        IDBFS.getLocalSet(mount, (err, local) => {
          if (err) return callback(err);
  
          IDBFS.getRemoteSet(mount, (err, remote) => {
            if (err) return callback(err);
  
            var src = populate ? remote : local;
            var dst = populate ? local : remote;
  
            IDBFS.reconcile(src, dst, callback);
          });
        });
      },quit:() => {
        Object.values(IDBFS.dbs).forEach((value) => value.close());
        IDBFS.dbs = {};
      },getDB:(name, callback) => {
        // check the cache first
        var db = IDBFS.dbs[name];
        if (db) {
          return callback(null, db);
        }
  
        var req;
        try {
          req = IDBFS.indexedDB().open(name, IDBFS.DB_VERSION);
        } catch (e) {
          return callback(e);
        }
        if (!req) {
          return callback("Unable to connect to IndexedDB");
        }
        req.onupgradeneeded = (e) => {
          var db = /** @type {IDBDatabase} */ (e.target.result);
          var transaction = e.target.transaction;
  
          var fileStore;
  
          if (db.objectStoreNames.contains(IDBFS.DB_STORE_NAME)) {
            fileStore = transaction.objectStore(IDBFS.DB_STORE_NAME);
          } else {
            fileStore = db.createObjectStore(IDBFS.DB_STORE_NAME);
          }
  
          if (!fileStore.indexNames.contains('timestamp')) {
            fileStore.createIndex('timestamp', 'timestamp', { unique: false });
          }
        };
        req.onsuccess = () => {
          db = /** @type {IDBDatabase} */ (req.result);
  
          // add to the cache
          IDBFS.dbs[name] = db;
          callback(null, db);
        };
        req.onerror = (e) => {
          callback(this.error);
          e.preventDefault();
        };
      },getLocalSet:(mount, callback) => {
        var entries = {};
  
        function isRealDir(p) {
          return p !== '.' && p !== '..';
        }        function toAbsolute(root) {
          return (p) => {
            return PATH.join2(root, p);
          }
        }  
        var check = FS.readdir(mount.mountpoint).filter(isRealDir).map(toAbsolute(mount.mountpoint));
  
        while (check.length) {
          var path = check.pop();
          var stat;
  
          try {
            stat = FS.stat(path);
          } catch (e) {
            return callback(e);
          }
  
          if (FS.isDir(stat.mode)) {
            check.push.apply(check, FS.readdir(path).filter(isRealDir).map(toAbsolute(path)));
          }
  
          entries[path] = { 'timestamp': stat.mtime };
        }
  
        return callback(null, { type: 'local', entries: entries });
      },getRemoteSet:(mount, callback) => {
        var entries = {};
  
        IDBFS.getDB(mount.mountpoint, (err, db) => {
          if (err) return callback(err);
  
          try {
            var transaction = db.transaction([IDBFS.DB_STORE_NAME], 'readonly');
            transaction.onerror = (e) => {
              callback(this.error);
              e.preventDefault();
            };
  
            var store = transaction.objectStore(IDBFS.DB_STORE_NAME);
            var index = store.index('timestamp');
  
            index.openKeyCursor().onsuccess = (event) => {
              var cursor = event.target.result;
  
              if (!cursor) {
                return callback(null, { type: 'remote', db: db, entries: entries });
              }
  
              entries[cursor.primaryKey] = { 'timestamp': cursor.key };
  
              cursor.continue();
            };
          } catch (e) {
            return callback(e);
          }
        });
      },loadLocalEntry:(path, callback) => {
        var stat, node;
  
        try {
          var lookup = FS.lookupPath(path);
          node = lookup.node;
          stat = FS.stat(path);
        } catch (e) {
          return callback(e);
        }
  
        if (FS.isDir(stat.mode)) {
          return callback(null, { 'timestamp': stat.mtime, 'mode': stat.mode });
        } else if (FS.isFile(stat.mode)) {
          // Performance consideration: storing a normal JavaScript array to a IndexedDB is much slower than storing a typed array.
          // Therefore always convert the file contents to a typed array first before writing the data to IndexedDB.
          node.contents = MEMFS.getFileDataAsTypedArray(node);
          return callback(null, { 'timestamp': stat.mtime, 'mode': stat.mode, 'contents': node.contents });
        } else {
          return callback(new Error('node type not supported'));
        }
      },storeLocalEntry:(path, entry, callback) => {
        try {
          if (FS.isDir(entry['mode'])) {
            FS.mkdirTree(path, entry['mode']);
          } else if (FS.isFile(entry['mode'])) {
            FS.writeFile(path, entry['contents'], { canOwn: true });
          } else {
            return callback(new Error('node type not supported'));
          }
  
          FS.chmod(path, entry['mode']);
          FS.utime(path, entry['timestamp'], entry['timestamp']);
        } catch (e) {
          return callback(e);
        }
  
        callback(null);
      },removeLocalEntry:(path, callback) => {
        try {
          var stat = FS.stat(path);
  
          if (FS.isDir(stat.mode)) {
            FS.rmdir(path);
          } else if (FS.isFile(stat.mode)) {
            FS.unlink(path);
          }
        } catch (e) {
          return callback(e);
        }
  
        callback(null);
      },loadRemoteEntry:(store, path, callback) => {
        var req = store.get(path);
        req.onsuccess = (event) => { callback(null, event.target.result); };
        req.onerror = (e) => {
          callback(this.error);
          e.preventDefault();
        };
      },storeRemoteEntry:(store, path, entry, callback) => {
        try {
          var req = store.put(entry, path);
        } catch (e) {
          callback(e);
          return;
        }
        req.onsuccess = () => { callback(null); };
        req.onerror = (e) => {
          callback(this.error);
          e.preventDefault();
        };
      },removeRemoteEntry:(store, path, callback) => {
        var req = store.delete(path);
        req.onsuccess = () => { callback(null); };
        req.onerror = (e) => {
          callback(this.error);
          e.preventDefault();
        };
      },reconcile:(src, dst, callback) => {
        var total = 0;
  
        var create = [];
        Object.keys(src.entries).forEach(function (key) {
          var e = src.entries[key];
          var e2 = dst.entries[key];
          if (!e2 || e['timestamp'].getTime() != e2['timestamp'].getTime()) {
            create.push(key);
            total++;
          }
        });
  
        var remove = [];
        Object.keys(dst.entries).forEach(function (key) {
          if (!src.entries[key]) {
            remove.push(key);
            total++;
          }
        });
  
        if (!total) {
          return callback(null);
        }
  
        var errored = false;
        var db = src.type === 'remote' ? src.db : dst.db;
        var transaction = db.transaction([IDBFS.DB_STORE_NAME], 'readwrite');
        var store = transaction.objectStore(IDBFS.DB_STORE_NAME);
  
        function done(err) {
          if (err && !errored) {
            errored = true;
            return callback(err);
          }
        }  
        transaction.onerror = (e) => {
          done(this.error);
          e.preventDefault();
        };
  
        transaction.oncomplete = (e) => {
          if (!errored) {
            callback(null);
          }
        };
  
        // sort paths in ascending order so directory entries are created
        // before the files inside them
        create.sort().forEach((path) => {
          if (dst.type === 'local') {
            IDBFS.loadRemoteEntry(store, path, (err, entry) => {
              if (err) return done(err);
              IDBFS.storeLocalEntry(path, entry, done);
            });
          } else {
            IDBFS.loadLocalEntry(path, (err, entry) => {
              if (err) return done(err);
              IDBFS.storeRemoteEntry(store, path, entry, done);
            });
          }
        });
  
        // sort paths in descending order so files are deleted before their
        // parent directories
        remove.sort().reverse().forEach((path) => {
          if (dst.type === 'local') {
            IDBFS.removeLocalEntry(path, done);
          } else {
            IDBFS.removeRemoteEntry(store, path, done);
          }
        });
      }};
  
  var ERRNO_MESSAGES = {0:"Success",1:"Arg list too long",2:"Permission denied",3:"Address already in use",4:"Address not available",5:"Address family not supported by protocol family",6:"No more processes",7:"Socket already connected",8:"Bad file number",9:"Trying to read unreadable message",10:"Mount device busy",11:"Operation canceled",12:"No children",13:"Connection aborted",14:"Connection refused",15:"Connection reset by peer",16:"File locking deadlock error",17:"Destination address required",18:"Math arg out of domain of func",19:"Quota exceeded",20:"File exists",21:"Bad address",22:"File too large",23:"Host is unreachable",24:"Identifier removed",25:"Illegal byte sequence",26:"Connection already in progress",27:"Interrupted system call",28:"Invalid argument",29:"I/O error",30:"Socket is already connected",31:"Is a directory",32:"Too many symbolic links",33:"Too many open files",34:"Too many links",35:"Message too long",36:"Multihop attempted",37:"File or path name too long",38:"Network interface is not configured",39:"Connection reset by network",40:"Network is unreachable",41:"Too many open files in system",42:"No buffer space available",43:"No such device",44:"No such file or directory",45:"Exec format error",46:"No record locks available",47:"The link has been severed",48:"Not enough core",49:"No message of desired type",50:"Protocol not available",51:"No space left on device",52:"Function not implemented",53:"Socket is not connected",54:"Not a directory",55:"Directory not empty",56:"State not recoverable",57:"Socket operation on non-socket",59:"Not a typewriter",60:"No such device or address",61:"Value too large for defined data type",62:"Previous owner died",63:"Not super-user",64:"Broken pipe",65:"Protocol error",66:"Unknown protocol",67:"Protocol wrong type for socket",68:"Math result not representable",69:"Read only file system",70:"Illegal seek",71:"No such process",72:"Stale file handle",73:"Connection timed out",74:"Text file busy",75:"Cross-device link",100:"Device not a stream",101:"Bad font file fmt",102:"Invalid slot",103:"Invalid request code",104:"No anode",105:"Block device required",106:"Channel number out of range",107:"Level 3 halted",108:"Level 3 reset",109:"Link number out of range",110:"Protocol driver not attached",111:"No CSI structure available",112:"Level 2 halted",113:"Invalid exchange",114:"Invalid request descriptor",115:"Exchange full",116:"No data (for no delay io)",117:"Timer expired",118:"Out of streams resources",119:"Machine is not on the network",120:"Package not installed",121:"The object is remote",122:"Advertise error",123:"Srmount error",124:"Communication error on send",125:"Cross mount point (not really error)",126:"Given log. name not unique",127:"f.d. invalid for this operation",128:"Remote address changed",129:"Can   access a needed shared lib",130:"Accessing a corrupted shared lib",131:".lib section in a.out corrupted",132:"Attempting to link in too many libs",133:"Attempting to exec a shared library",135:"Streams pipe error",136:"Too many users",137:"Socket type not supported",138:"Not supported",139:"Protocol family not supported",140:"Can't send after socket shutdown",141:"Too many references",142:"Host is down",148:"No medium (in tape drive)",156:"Level 2 not synchronized"};
  
  var ERRNO_CODES = {};
  
  var FS = {root:null,mounts:[],devices:{},streams:[],nextInode:1,nameTable:null,currentPath:"/",initialized:false,ignorePermissions:true,ErrnoError:null,genericErrors:{},filesystems:null,syncFSRequests:0,lookupPath:(path, opts = {}) => {
        path = PATH_FS.resolve(path);
  
        if (!path) return { path: '', node: null };
  
        var defaults = {
          follow_mount: true,
          recurse_count: 0
        };
        opts = Object.assign(defaults, opts);
  
        if (opts.recurse_count > 8) {  // max recursive lookup of 8
          throw new FS.ErrnoError(32);
        }
  
        // split the absolute path
        var parts = path.split('/').filter((p) => !!p);
  
        // start at the root
        var current = FS.root;
        var current_path = '/';
  
        for (var i = 0; i < parts.length; i++) {
          var islast = (i === parts.length-1);
          if (islast && opts.parent) {
            // stop resolving
            break;
          }
  
          current = FS.lookupNode(current, parts[i]);
          current_path = PATH.join2(current_path, parts[i]);
  
          // jump to the mount's root node if this is a mountpoint
          if (FS.isMountpoint(current)) {
            if (!islast || (islast && opts.follow_mount)) {
              current = current.mounted.root;
            }
          }
  
          // by default, lookupPath will not follow a symlink if it is the final path component.
          // setting opts.follow = true will override this behavior.
          if (!islast || opts.follow) {
            var count = 0;
            while (FS.isLink(current.mode)) {
              var link = FS.readlink(current_path);
              current_path = PATH_FS.resolve(PATH.dirname(current_path), link);
  
              var lookup = FS.lookupPath(current_path, { recurse_count: opts.recurse_count + 1 });
              current = lookup.node;
  
              if (count++ > 40) {  // limit max consecutive symlinks to 40 (SYMLOOP_MAX).
                throw new FS.ErrnoError(32);
              }
            }
          }
        }
  
        return { path: current_path, node: current };
      },getPath:(node) => {
        var path;
        while (true) {
          if (FS.isRoot(node)) {
            var mount = node.mount.mountpoint;
            if (!path) return mount;
            return mount[mount.length-1] !== '/' ? mount + '/' + path : mount + path;
          }
          path = path ? node.name + '/' + path : node.name;
          node = node.parent;
        }
      },hashName:(parentid, name) => {
        var hash = 0;
  
        for (var i = 0; i < name.length; i++) {
          hash = ((hash << 5) - hash + name.charCodeAt(i)) | 0;
        }
        return ((parentid + hash) >>> 0) % FS.nameTable.length;
      },hashAddNode:(node) => {
        var hash = FS.hashName(node.parent.id, node.name);
        node.name_next = FS.nameTable[hash];
        FS.nameTable[hash] = node;
      },hashRemoveNode:(node) => {
        var hash = FS.hashName(node.parent.id, node.name);
        if (FS.nameTable[hash] === node) {
          FS.nameTable[hash] = node.name_next;
        } else {
          var current = FS.nameTable[hash];
          while (current) {
            if (current.name_next === node) {
              current.name_next = node.name_next;
              break;
            }
            current = current.name_next;
          }
        }
      },lookupNode:(parent, name) => {
        var errCode = FS.mayLookup(parent);
        if (errCode) {
          throw new FS.ErrnoError(errCode, parent);
        }
        var hash = FS.hashName(parent.id, name);
        for (var node = FS.nameTable[hash]; node; node = node.name_next) {
          var nodeName = node.name;
          if (node.parent.id === parent.id && nodeName === name) {
            return node;
          }
        }
        // if we failed to find it in the cache, call into the VFS
        return FS.lookup(parent, name);
      },createNode:(parent, name, mode, rdev) => {
        assert(typeof parent == 'object');
        var node = new FS.FSNode(parent, name, mode, rdev);
  
        FS.hashAddNode(node);
  
        return node;
      },destroyNode:(node) => {
        FS.hashRemoveNode(node);
      },isRoot:(node) => {
        return node === node.parent;
      },isMountpoint:(node) => {
        return !!node.mounted;
      },isFile:(mode) => {
        return (mode & 61440) === 32768;
      },isDir:(mode) => {
        return (mode & 61440) === 16384;
      },isLink:(mode) => {
        return (mode & 61440) === 40960;
      },isChrdev:(mode) => {
        return (mode & 61440) === 8192;
      },isBlkdev:(mode) => {
        return (mode & 61440) === 24576;
      },isFIFO:(mode) => {
        return (mode & 61440) === 4096;
      },isSocket:(mode) => {
        return (mode & 49152) === 49152;
      },flagModes:{"r":0,"r+":2,"w":577,"w+":578,"a":1089,"a+":1090},modeStringToFlags:(str) => {
        var flags = FS.flagModes[str];
        if (typeof flags == 'undefined') {
          throw new Error('Unknown file open mode: ' + str);
        }
        return flags;
      },flagsToPermissionString:(flag) => {
        var perms = ['r', 'w', 'rw'][flag & 3];
        if ((flag & 512)) {
          perms += 'w';
        }
        return perms;
      },nodePermissions:(node, perms) => {
        if (FS.ignorePermissions) {
          return 0;
        }
        // return 0 if any user, group or owner bits are set.
        if (perms.includes('r') && !(node.mode & 292)) {
          return 2;
        } else if (perms.includes('w') && !(node.mode & 146)) {
          return 2;
        } else if (perms.includes('x') && !(node.mode & 73)) {
          return 2;
        }
        return 0;
      },mayLookup:(dir) => {
        var errCode = FS.nodePermissions(dir, 'x');
        if (errCode) return errCode;
        if (!dir.node_ops.lookup) return 2;
        return 0;
      },mayCreate:(dir, name) => {
        try {
          var node = FS.lookupNode(dir, name);
          return 20;
        } catch (e) {
        }
        return FS.nodePermissions(dir, 'wx');
      },mayDelete:(dir, name, isdir) => {
        var node;
        try {
          node = FS.lookupNode(dir, name);
        } catch (e) {
          return e.errno;
        }
        var errCode = FS.nodePermissions(dir, 'wx');
        if (errCode) {
          return errCode;
        }
        if (isdir) {
          if (!FS.isDir(node.mode)) {
            return 54;
          }
          if (FS.isRoot(node) || FS.getPath(node) === FS.cwd()) {
            return 10;
          }
        } else {
          if (FS.isDir(node.mode)) {
            return 31;
          }
        }
        return 0;
      },mayOpen:(node, flags) => {
        if (!node) {
          return 44;
        }
        if (FS.isLink(node.mode)) {
          return 32;
        } else if (FS.isDir(node.mode)) {
          if (FS.flagsToPermissionString(flags) !== 'r' || // opening for write
              (flags & 512)) { // TODO: check for O_SEARCH? (== search for dir only)
            return 31;
          }
        }
        return FS.nodePermissions(node, FS.flagsToPermissionString(flags));
      },MAX_OPEN_FDS:4096,nextfd:(fd_start = 0, fd_end = FS.MAX_OPEN_FDS) => {
        for (var fd = fd_start; fd <= fd_end; fd++) {
          if (!FS.streams[fd]) {
            return fd;
          }
        }
        throw new FS.ErrnoError(33);
      },getStream:(fd) => FS.streams[fd],createStream:(stream, fd_start, fd_end) => {
        if (!FS.FSStream) {
          FS.FSStream = /** @constructor */ function() {
            this.shared = { };
          };
          FS.FSStream.prototype = {};
          Object.defineProperties(FS.FSStream.prototype, {
            object: {
              /** @this {FS.FSStream} */
              get: function() { return this.node; },
              /** @this {FS.FSStream} */
              set: function(val) { this.node = val; }
            },
            isRead: {
              /** @this {FS.FSStream} */
              get: function() { return (this.flags & 2097155) !== 1; }
            },
            isWrite: {
              /** @this {FS.FSStream} */
              get: function() { return (this.flags & 2097155) !== 0; }
            },
            isAppend: {
              /** @this {FS.FSStream} */
              get: function() { return (this.flags & 1024); }
            },
            flags: {
              /** @this {FS.FSStream} */
              get: function() { return this.shared.flags; },
              /** @this {FS.FSStream} */
              set: function(val) { this.shared.flags = val; },
            },
            position : {
              /** @this {FS.FSStream} */
              get: function() { return this.shared.position; },
              /** @this {FS.FSStream} */
              set: function(val) { this.shared.position = val; },
            },
          });
        }
        // clone it, so we can return an instance of FSStream
        stream = Object.assign(new FS.FSStream(), stream);
        var fd = FS.nextfd(fd_start, fd_end);
        stream.fd = fd;
        FS.streams[fd] = stream;
        return stream;
      },closeStream:(fd) => {
        FS.streams[fd] = null;
      },chrdev_stream_ops:{open:(stream) => {
          var device = FS.getDevice(stream.node.rdev);
          // override node's stream ops with the device's
          stream.stream_ops = device.stream_ops;
          // forward the open call
          if (stream.stream_ops.open) {
            stream.stream_ops.open(stream);
          }
        },llseek:() => {
          throw new FS.ErrnoError(70);
        }},major:(dev) => ((dev) >> 8),minor:(dev) => ((dev) & 0xff),makedev:(ma, mi) => ((ma) << 8 | (mi)),registerDevice:(dev, ops) => {
        FS.devices[dev] = { stream_ops: ops };
      },getDevice:(dev) => FS.devices[dev],getMounts:(mount) => {
        var mounts = [];
        var check = [mount];
  
        while (check.length) {
          var m = check.pop();
  
          mounts.push(m);
  
          check.push.apply(check, m.mounts);
        }
  
        return mounts;
      },syncfs:(populate, callback) => {
        if (typeof populate == 'function') {
          callback = populate;
          populate = false;
        }
  
        FS.syncFSRequests++;
  
        if (FS.syncFSRequests > 1) {
          err('warning: ' + FS.syncFSRequests + ' FS.syncfs operations in flight at once, probably just doing extra work');
        }
  
        var mounts = FS.getMounts(FS.root.mount);
        var completed = 0;
  
        function doCallback(errCode) {
          assert(FS.syncFSRequests > 0);
          FS.syncFSRequests--;
          return callback(errCode);
        }
  
        function done(errCode) {
          if (errCode) {
            if (!done.errored) {
              done.errored = true;
              return doCallback(errCode);
            }
            return;
          }
          if (++completed >= mounts.length) {
            doCallback(null);
          }
        }  
        // sync all mounts
        mounts.forEach((mount) => {
          if (!mount.type.syncfs) {
            return done(null);
          }
          mount.type.syncfs(mount, populate, done);
        });
      },mount:(type, opts, mountpoint) => {
        if (typeof type == 'string') {
          // The filesystem was not included, and instead we have an error
          // message stored in the variable.
          throw type;
        }
        var root = mountpoint === '/';
        var pseudo = !mountpoint;
        var node;
  
        if (root && FS.root) {
          throw new FS.ErrnoError(10);
        } else if (!root && !pseudo) {
          var lookup = FS.lookupPath(mountpoint, { follow_mount: false });
  
          mountpoint = lookup.path;  // use the absolute path
          node = lookup.node;
  
          if (FS.isMountpoint(node)) {
            throw new FS.ErrnoError(10);
          }
  
          if (!FS.isDir(node.mode)) {
            throw new FS.ErrnoError(54);
          }
        }
  
        var mount = {
          type: type,
          opts: opts,
          mountpoint: mountpoint,
          mounts: []
        };
  
        // create a root node for the fs
        var mountRoot = type.mount(mount);
        mountRoot.mount = mount;
        mount.root = mountRoot;
  
        if (root) {
          FS.root = mountRoot;
        } else if (node) {
          // set as a mountpoint
          node.mounted = mount;
  
          // add the new mount to the current mount's children
          if (node.mount) {
            node.mount.mounts.push(mount);
          }
        }
  
        return mountRoot;
      },unmount:(mountpoint) => {
        var lookup = FS.lookupPath(mountpoint, { follow_mount: false });
  
        if (!FS.isMountpoint(lookup.node)) {
          throw new FS.ErrnoError(28);
        }
  
        // destroy the nodes for this mount, and all its child mounts
        var node = lookup.node;
        var mount = node.mounted;
        var mounts = FS.getMounts(mount);
  
        Object.keys(FS.nameTable).forEach((hash) => {
          var current = FS.nameTable[hash];
  
          while (current) {
            var next = current.name_next;
  
            if (mounts.includes(current.mount)) {
              FS.destroyNode(current);
            }
  
            current = next;
          }
        });
  
        // no longer a mountpoint
        node.mounted = null;
  
        // remove this mount from the child mounts
        var idx = node.mount.mounts.indexOf(mount);
        assert(idx !== -1);
        node.mount.mounts.splice(idx, 1);
      },lookup:(parent, name) => {
        return parent.node_ops.lookup(parent, name);
      },mknod:(path, mode, dev) => {
        var lookup = FS.lookupPath(path, { parent: true });
        var parent = lookup.node;
        var name = PATH.basename(path);
        if (!name || name === '.' || name === '..') {
          throw new FS.ErrnoError(28);
        }
        var errCode = FS.mayCreate(parent, name);
        if (errCode) {
          throw new FS.ErrnoError(errCode);
        }
        if (!parent.node_ops.mknod) {
          throw new FS.ErrnoError(63);
        }
        return parent.node_ops.mknod(parent, name, mode, dev);
      },create:(path, mode) => {
        mode = mode !== undefined ? mode : 438 /* 0666 */;
        mode &= 4095;
        mode |= 32768;
        return FS.mknod(path, mode, 0);
      },mkdir:(path, mode) => {
        mode = mode !== undefined ? mode : 511 /* 0777 */;
        mode &= 511 | 512;
        mode |= 16384;
        return FS.mknod(path, mode, 0);
      },mkdirTree:(path, mode) => {
        var dirs = path.split('/');
        var d = '';
        for (var i = 0; i < dirs.length; ++i) {
          if (!dirs[i]) continue;
          d += '/' + dirs[i];
          try {
            FS.mkdir(d, mode);
          } catch(e) {
            if (e.errno != 20) throw e;
          }
        }
      },mkdev:(path, mode, dev) => {
        if (typeof dev == 'undefined') {
          dev = mode;
          mode = 438 /* 0666 */;
        }
        mode |= 8192;
        return FS.mknod(path, mode, dev);
      },symlink:(oldpath, newpath) => {
        if (!PATH_FS.resolve(oldpath)) {
          throw new FS.ErrnoError(44);
        }
        var lookup = FS.lookupPath(newpath, { parent: true });
        var parent = lookup.node;
        if (!parent) {
          throw new FS.ErrnoError(44);
        }
        var newname = PATH.basename(newpath);
        var errCode = FS.mayCreate(parent, newname);
        if (errCode) {
          throw new FS.ErrnoError(errCode);
        }
        if (!parent.node_ops.symlink) {
          throw new FS.ErrnoError(63);
        }
        return parent.node_ops.symlink(parent, newname, oldpath);
      },rename:(old_path, new_path) => {
        var old_dirname = PATH.dirname(old_path);
        var new_dirname = PATH.dirname(new_path);
        var old_name = PATH.basename(old_path);
        var new_name = PATH.basename(new_path);
        // parents must exist
        var lookup, old_dir, new_dir;
  
        // let the errors from non existant directories percolate up
        lookup = FS.lookupPath(old_path, { parent: true });
        old_dir = lookup.node;
        lookup = FS.lookupPath(new_path, { parent: true });
        new_dir = lookup.node;
  
        if (!old_dir || !new_dir) throw new FS.ErrnoError(44);
        // need to be part of the same mount
        if (old_dir.mount !== new_dir.mount) {
          throw new FS.ErrnoError(75);
        }
        // source must exist
        var old_node = FS.lookupNode(old_dir, old_name);
        // old path should not be an ancestor of the new path
        var relative = PATH_FS.relative(old_path, new_dirname);
        if (relative.charAt(0) !== '.') {
          throw new FS.ErrnoError(28);
        }
        // new path should not be an ancestor of the old path
        relative = PATH_FS.relative(new_path, old_dirname);
        if (relative.charAt(0) !== '.') {
          throw new FS.ErrnoError(55);
        }
        // see if the new path already exists
        var new_node;
        try {
          new_node = FS.lookupNode(new_dir, new_name);
        } catch (e) {
          // not fatal
        }
        // early out if nothing needs to change
        if (old_node === new_node) {
          return;
        }
        // we'll need to delete the old entry
        var isdir = FS.isDir(old_node.mode);
        var errCode = FS.mayDelete(old_dir, old_name, isdir);
        if (errCode) {
          throw new FS.ErrnoError(errCode);
        }
        // need delete permissions if we'll be overwriting.
        // need create permissions if new doesn't already exist.
        errCode = new_node ?
          FS.mayDelete(new_dir, new_name, isdir) :
          FS.mayCreate(new_dir, new_name);
        if (errCode) {
          throw new FS.ErrnoError(errCode);
        }
        if (!old_dir.node_ops.rename) {
          throw new FS.ErrnoError(63);
        }
        if (FS.isMountpoint(old_node) || (new_node && FS.isMountpoint(new_node))) {
          throw new FS.ErrnoError(10);
        }
        // if we are going to change the parent, check write permissions
        if (new_dir !== old_dir) {
          errCode = FS.nodePermissions(old_dir, 'w');
          if (errCode) {
            throw new FS.ErrnoError(errCode);
          }
        }
        // remove the node from the lookup hash
        FS.hashRemoveNode(old_node);
        // do the underlying fs rename
        try {
          old_dir.node_ops.rename(old_node, new_dir, new_name);
        } catch (e) {
          throw e;
        } finally {
          // add the node back to the hash (in case node_ops.rename
          // changed its name)
          FS.hashAddNode(old_node);
        }
      },rmdir:(path) => {
        var lookup = FS.lookupPath(path, { parent: true });
        var parent = lookup.node;
        var name = PATH.basename(path);
        var node = FS.lookupNode(parent, name);
        var errCode = FS.mayDelete(parent, name, true);
        if (errCode) {
          throw new FS.ErrnoError(errCode);
        }
        if (!parent.node_ops.rmdir) {
          throw new FS.ErrnoError(63);
        }
        if (FS.isMountpoint(node)) {
          throw new FS.ErrnoError(10);
        }
        parent.node_ops.rmdir(parent, name);
        FS.destroyNode(node);
      },readdir:(path) => {
        var lookup = FS.lookupPath(path, { follow: true });
        var node = lookup.node;
        if (!node.node_ops.readdir) {
          throw new FS.ErrnoError(54);
        }
        return node.node_ops.readdir(node);
      },unlink:(path) => {
        var lookup = FS.lookupPath(path, { parent: true });
        var parent = lookup.node;
        if (!parent) {
          throw new FS.ErrnoError(44);
        }
        var name = PATH.basename(path);
        var node = FS.lookupNode(parent, name);
        var errCode = FS.mayDelete(parent, name, false);
        if (errCode) {
          // According to POSIX, we should map EISDIR to EPERM, but
          // we instead do what Linux does (and we must, as we use
          // the musl linux libc).
          throw new FS.ErrnoError(errCode);
        }
        if (!parent.node_ops.unlink) {
          throw new FS.ErrnoError(63);
        }
        if (FS.isMountpoint(node)) {
          throw new FS.ErrnoError(10);
        }
        parent.node_ops.unlink(parent, name);
        FS.destroyNode(node);
      },readlink:(path) => {
        var lookup = FS.lookupPath(path);
        var link = lookup.node;
        if (!link) {
          throw new FS.ErrnoError(44);
        }
        if (!link.node_ops.readlink) {
          throw new FS.ErrnoError(28);
        }
        return PATH_FS.resolve(FS.getPath(link.parent), link.node_ops.readlink(link));
      },stat:(path, dontFollow) => {
        var lookup = FS.lookupPath(path, { follow: !dontFollow });
        var node = lookup.node;
        if (!node) {
          throw new FS.ErrnoError(44);
        }
        if (!node.node_ops.getattr) {
          throw new FS.ErrnoError(63);
        }
        return node.node_ops.getattr(node);
      },lstat:(path) => {
        return FS.stat(path, true);
      },chmod:(path, mode, dontFollow) => {
        var node;
        if (typeof path == 'string') {
          var lookup = FS.lookupPath(path, { follow: !dontFollow });
          node = lookup.node;
        } else {
          node = path;
        }
        if (!node.node_ops.setattr) {
          throw new FS.ErrnoError(63);
        }
        node.node_ops.setattr(node, {
          mode: (mode & 4095) | (node.mode & ~4095),
          timestamp: Date.now()
        });
      },lchmod:(path, mode) => {
        FS.chmod(path, mode, true);
      },fchmod:(fd, mode) => {
        var stream = FS.getStream(fd);
        if (!stream) {
          throw new FS.ErrnoError(8);
        }
        FS.chmod(stream.node, mode);
      },chown:(path, uid, gid, dontFollow) => {
        var node;
        if (typeof path == 'string') {
          var lookup = FS.lookupPath(path, { follow: !dontFollow });
          node = lookup.node;
        } else {
          node = path;
        }
        if (!node.node_ops.setattr) {
          throw new FS.ErrnoError(63);
        }
        node.node_ops.setattr(node, {
          timestamp: Date.now()
          // we ignore the uid / gid for now
        });
      },lchown:(path, uid, gid) => {
        FS.chown(path, uid, gid, true);
      },fchown:(fd, uid, gid) => {
        var stream = FS.getStream(fd);
        if (!stream) {
          throw new FS.ErrnoError(8);
        }
        FS.chown(stream.node, uid, gid);
      },truncate:(path, len) => {
        if (len < 0) {
          throw new FS.ErrnoError(28);
        }
        var node;
        if (typeof path == 'string') {
          var lookup = FS.lookupPath(path, { follow: true });
          node = lookup.node;
        } else {
          node = path;
        }
        if (!node.node_ops.setattr) {
          throw new FS.ErrnoError(63);
        }
        if (FS.isDir(node.mode)) {
          throw new FS.ErrnoError(31);
        }
        if (!FS.isFile(node.mode)) {
          throw new FS.ErrnoError(28);
        }
        var errCode = FS.nodePermissions(node, 'w');
        if (errCode) {
          throw new FS.ErrnoError(errCode);
        }
        node.node_ops.setattr(node, {
          size: len,
          timestamp: Date.now()
        });
      },ftruncate:(fd, len) => {
        var stream = FS.getStream(fd);
        if (!stream) {
          throw new FS.ErrnoError(8);
        }
        if ((stream.flags & 2097155) === 0) {
          throw new FS.ErrnoError(28);
        }
        FS.truncate(stream.node, len);
      },utime:(path, atime, mtime) => {
        var lookup = FS.lookupPath(path, { follow: true });
        var node = lookup.node;
        node.node_ops.setattr(node, {
          timestamp: Math.max(atime, mtime)
        });
      },open:(path, flags, mode) => {
        if (path === "") {
          throw new FS.ErrnoError(44);
        }
        flags = typeof flags == 'string' ? FS.modeStringToFlags(flags) : flags;
        mode = typeof mode == 'undefined' ? 438 /* 0666 */ : mode;
        if ((flags & 64)) {
          mode = (mode & 4095) | 32768;
        } else {
          mode = 0;
        }
        var node;
        if (typeof path == 'object') {
          node = path;
        } else {
          path = PATH.normalize(path);
          try {
            var lookup = FS.lookupPath(path, {
              follow: !(flags & 131072)
            });
            node = lookup.node;
          } catch (e) {
            // ignore
          }
        }
        // perhaps we need to create the node
        var created = false;
        if ((flags & 64)) {
          if (node) {
            // if O_CREAT and O_EXCL are set, error out if the node already exists
            if ((flags & 128)) {
              throw new FS.ErrnoError(20);
            }
          } else {
            // node doesn't exist, try to create it
            node = FS.mknod(path, mode, 0);
            created = true;
          }
        }
        if (!node) {
          throw new FS.ErrnoError(44);
        }
        // can't truncate a device
        if (FS.isChrdev(node.mode)) {
          flags &= ~512;
        }
        // if asked only for a directory, then this must be one
        if ((flags & 65536) && !FS.isDir(node.mode)) {
          throw new FS.ErrnoError(54);
        }
        // check permissions, if this is not a file we just created now (it is ok to
        // create and write to a file with read-only permissions; it is read-only
        // for later use)
        if (!created) {
          var errCode = FS.mayOpen(node, flags);
          if (errCode) {
            throw new FS.ErrnoError(errCode);
          }
        }
        // do truncation if necessary
        if ((flags & 512) && !created) {
          FS.truncate(node, 0);
        }
        // we've already handled these, don't pass down to the underlying vfs
        flags &= ~(128 | 512 | 131072);
  
        // register the stream with the filesystem
        var stream = FS.createStream({
          node: node,
          path: FS.getPath(node),  // we want the absolute path to the node
          flags: flags,
          seekable: true,
          position: 0,
          stream_ops: node.stream_ops,
          // used by the file family libc calls (fopen, fwrite, ferror, etc.)
          ungotten: [],
          error: false
        });
        // call the new stream's open function
        if (stream.stream_ops.open) {
          stream.stream_ops.open(stream);
        }
        if (Module['logReadFiles'] && !(flags & 1)) {
          if (!FS.readFiles) FS.readFiles = {};
          if (!(path in FS.readFiles)) {
            FS.readFiles[path] = 1;
          }
        }
        return stream;
      },close:(stream) => {
        if (FS.isClosed(stream)) {
          throw new FS.ErrnoError(8);
        }
        if (stream.getdents) stream.getdents = null; // free readdir state
        try {
          if (stream.stream_ops.close) {
            stream.stream_ops.close(stream);
          }
        } catch (e) {
          throw e;
        } finally {
          FS.closeStream(stream.fd);
        }
        stream.fd = null;
      },isClosed:(stream) => {
        return stream.fd === null;
      },llseek:(stream, offset, whence) => {
        if (FS.isClosed(stream)) {
          throw new FS.ErrnoError(8);
        }
        if (!stream.seekable || !stream.stream_ops.llseek) {
          throw new FS.ErrnoError(70);
        }
        if (whence != 0 && whence != 1 && whence != 2) {
          throw new FS.ErrnoError(28);
        }
        stream.position = stream.stream_ops.llseek(stream, offset, whence);
        stream.ungotten = [];
        return stream.position;
      },read:(stream, buffer, offset, length, position) => {
        if (length < 0 || position < 0) {
          throw new FS.ErrnoError(28);
        }
        if (FS.isClosed(stream)) {
          throw new FS.ErrnoError(8);
        }
        if ((stream.flags & 2097155) === 1) {
          throw new FS.ErrnoError(8);
        }
        if (FS.isDir(stream.node.mode)) {
          throw new FS.ErrnoError(31);
        }
        if (!stream.stream_ops.read) {
          throw new FS.ErrnoError(28);
        }
        var seeking = typeof position != 'undefined';
        if (!seeking) {
          position = stream.position;
        } else if (!stream.seekable) {
          throw new FS.ErrnoError(70);
        }
        var bytesRead = stream.stream_ops.read(stream, buffer, offset, length, position);
        if (!seeking) stream.position += bytesRead;
        return bytesRead;
      },write:(stream, buffer, offset, length, position, canOwn) => {
        if (length < 0 || position < 0) {
          throw new FS.ErrnoError(28);
        }
        if (FS.isClosed(stream)) {
          throw new FS.ErrnoError(8);
        }
        if ((stream.flags & 2097155) === 0) {
          throw new FS.ErrnoError(8);
        }
        if (FS.isDir(stream.node.mode)) {
          throw new FS.ErrnoError(31);
        }
        if (!stream.stream_ops.write) {
          throw new FS.ErrnoError(28);
        }
        if (stream.seekable && stream.flags & 1024) {
          // seek to the end before writing in append mode
          FS.llseek(stream, 0, 2);
        }
        var seeking = typeof position != 'undefined';
        if (!seeking) {
          position = stream.position;
        } else if (!stream.seekable) {
          throw new FS.ErrnoError(70);
        }
        var bytesWritten = stream.stream_ops.write(stream, buffer, offset, length, position, canOwn);
        if (!seeking) stream.position += bytesWritten;
        return bytesWritten;
      },allocate:(stream, offset, length) => {
        if (FS.isClosed(stream)) {
          throw new FS.ErrnoError(8);
        }
        if (offset < 0 || length <= 0) {
          throw new FS.ErrnoError(28);
        }
        if ((stream.flags & 2097155) === 0) {
          throw new FS.ErrnoError(8);
        }
        if (!FS.isFile(stream.node.mode) && !FS.isDir(stream.node.mode)) {
          throw new FS.ErrnoError(43);
        }
        if (!stream.stream_ops.allocate) {
          throw new FS.ErrnoError(138);
        }
        stream.stream_ops.allocate(stream, offset, length);
      },mmap:(stream, length, position, prot, flags) => {
        // User requests writing to file (prot & PROT_WRITE != 0).
        // Checking if we have permissions to write to the file unless
        // MAP_PRIVATE flag is set. According to POSIX spec it is possible
        // to write to file opened in read-only mode with MAP_PRIVATE flag,
        // as all modifications will be visible only in the memory of
        // the current process.
        if ((prot & 2) !== 0
            && (flags & 2) === 0
            && (stream.flags & 2097155) !== 2) {
          throw new FS.ErrnoError(2);
        }
        if ((stream.flags & 2097155) === 1) {
          throw new FS.ErrnoError(2);
        }
        if (!stream.stream_ops.mmap) {
          throw new FS.ErrnoError(43);
        }
        return stream.stream_ops.mmap(stream, length, position, prot, flags);
      },msync:(stream, buffer, offset, length, mmapFlags) => {
        if (!stream.stream_ops.msync) {
          return 0;
        }
        return stream.stream_ops.msync(stream, buffer, offset, length, mmapFlags);
      },munmap:(stream) => 0,ioctl:(stream, cmd, arg) => {
        if (!stream.stream_ops.ioctl) {
          throw new FS.ErrnoError(59);
        }
        return stream.stream_ops.ioctl(stream, cmd, arg);
      },readFile:(path, opts = {}) => {
        opts.flags = opts.flags || 0;
        opts.encoding = opts.encoding || 'binary';
        if (opts.encoding !== 'utf8' && opts.encoding !== 'binary') {
          throw new Error('Invalid encoding type "' + opts.encoding + '"');
        }
        var ret;
        var stream = FS.open(path, opts.flags);
        var stat = FS.stat(path);
        var length = stat.size;
        var buf = new Uint8Array(length);
        FS.read(stream, buf, 0, length, 0);
        if (opts.encoding === 'utf8') {
          ret = UTF8ArrayToString(buf, 0);
        } else if (opts.encoding === 'binary') {
          ret = buf;
        }
        FS.close(stream);
        return ret;
      },writeFile:(path, data, opts = {}) => {
        opts.flags = opts.flags || 577;
        var stream = FS.open(path, opts.flags, opts.mode);
        if (typeof data == 'string') {
          var buf = new Uint8Array(lengthBytesUTF8(data)+1);
          var actualNumBytes = stringToUTF8Array(data, buf, 0, buf.length);
          FS.write(stream, buf, 0, actualNumBytes, undefined, opts.canOwn);
        } else if (ArrayBuffer.isView(data)) {
          FS.write(stream, data, 0, data.byteLength, undefined, opts.canOwn);
        } else {
          throw new Error('Unsupported data type');
        }
        FS.close(stream);
      },cwd:() => FS.currentPath,chdir:(path) => {
        var lookup = FS.lookupPath(path, { follow: true });
        if (lookup.node === null) {
          throw new FS.ErrnoError(44);
        }
        if (!FS.isDir(lookup.node.mode)) {
          throw new FS.ErrnoError(54);
        }
        var errCode = FS.nodePermissions(lookup.node, 'x');
        if (errCode) {
          throw new FS.ErrnoError(errCode);
        }
        FS.currentPath = lookup.path;
      },createDefaultDirectories:() => {
        FS.mkdir('/tmp');
        FS.mkdir('/home');
        FS.mkdir('/home/web_user');
      },createDefaultDevices:() => {
        // create /dev
        FS.mkdir('/dev');
        // setup /dev/null
        FS.registerDevice(FS.makedev(1, 3), {
          read: () => 0,
          write: (stream, buffer, offset, length, pos) => length,
        });
        FS.mkdev('/dev/null', FS.makedev(1, 3));
        // setup /dev/tty and /dev/tty1
        // stderr needs to print output using err() rather than out()
        // so we register a second tty just for it.
        TTY.register(FS.makedev(5, 0), TTY.default_tty_ops);
        TTY.register(FS.makedev(6, 0), TTY.default_tty1_ops);
        FS.mkdev('/dev/tty', FS.makedev(5, 0));
        FS.mkdev('/dev/tty1', FS.makedev(6, 0));
        // setup /dev/[u]random
        // use a buffer to avoid overhead of individual crypto calls per byte
        var randomBuffer = new Uint8Array(1024), randomLeft = 0;
        var randomByte = () => {
          if (randomLeft === 0) {
            randomLeft = randomFill(randomBuffer).byteLength;
          }
          return randomBuffer[--randomLeft];
        };
        FS.createDevice('/dev', 'random', randomByte);
        FS.createDevice('/dev', 'urandom', randomByte);
        // we're not going to emulate the actual shm device,
        // just create the tmp dirs that reside in it commonly
        FS.mkdir('/dev/shm');
        FS.mkdir('/dev/shm/tmp');
      },createSpecialDirectories:() => {
        // create /proc/self/fd which allows /proc/self/fd/6 => readlink gives the
        // name of the stream for fd 6 (see test_unistd_ttyname)
        FS.mkdir('/proc');
        var proc_self = FS.mkdir('/proc/self');
        FS.mkdir('/proc/self/fd');
        FS.mount({
          mount: () => {
            var node = FS.createNode(proc_self, 'fd', 16384 | 511 /* 0777 */, 73);
            node.node_ops = {
              lookup: (parent, name) => {
                var fd = +name;
                var stream = FS.getStream(fd);
                if (!stream) throw new FS.ErrnoError(8);
                var ret = {
                  parent: null,
                  mount: { mountpoint: 'fake' },
                  node_ops: { readlink: () => stream.path },
                };
                ret.parent = ret; // make it look like a simple root node
                return ret;
              }
            };
            return node;
          }
        }, {}, '/proc/self/fd');
      },createStandardStreams:() => {
        // TODO deprecate the old functionality of a single
        // input / output callback and that utilizes FS.createDevice
        // and instead require a unique set of stream ops
  
        // by default, we symlink the standard streams to the
        // default tty devices. however, if the standard streams
        // have been overwritten we create a unique device for
        // them instead.
        if (Module['stdin']) {
          FS.createDevice('/dev', 'stdin', Module['stdin']);
        } else {
          FS.symlink('/dev/tty', '/dev/stdin');
        }
        if (Module['stdout']) {
          FS.createDevice('/dev', 'stdout', null, Module['stdout']);
        } else {
          FS.symlink('/dev/tty', '/dev/stdout');
        }
        if (Module['stderr']) {
          FS.createDevice('/dev', 'stderr', null, Module['stderr']);
        } else {
          FS.symlink('/dev/tty1', '/dev/stderr');
        }
  
        // open default streams for the stdin, stdout and stderr devices
        var stdin = FS.open('/dev/stdin', 0);
        var stdout = FS.open('/dev/stdout', 1);
        var stderr = FS.open('/dev/stderr', 1);
        assert(stdin.fd === 0, 'invalid handle for stdin (' + stdin.fd + ')');
        assert(stdout.fd === 1, 'invalid handle for stdout (' + stdout.fd + ')');
        assert(stderr.fd === 2, 'invalid handle for stderr (' + stderr.fd + ')');
      },ensureErrnoError:() => {
        if (FS.ErrnoError) return;
        FS.ErrnoError = /** @this{Object} */ function ErrnoError(errno, node) {
          // We set the `name` property to be able to identify `FS.ErrnoError`
          // - the `name` is a standard ECMA-262 property of error objects. Kind of good to have it anyway.
          // - when using PROXYFS, an error can come from an underlying FS
          // as different FS objects have their own FS.ErrnoError each,
          // the test `err instanceof FS.ErrnoError` won't detect an error coming from another filesystem, causing bugs.
          // we'll use the reliable test `err.name == "ErrnoError"` instead
          this.name = 'ErrnoError';
          this.node = node;
          this.setErrno = /** @this{Object} */ function(errno) {
            this.errno = errno;
            for (var key in ERRNO_CODES) {
              if (ERRNO_CODES[key] === errno) {
                this.code = key;
                break;
              }
            }
          };
          this.setErrno(errno);
          this.message = ERRNO_MESSAGES[errno];
  
          // Try to get a maximally helpful stack trace. On Node.js, getting Error.stack
          // now ensures it shows what we want.
          if (this.stack) {
            // Define the stack property for Node.js 4, which otherwise errors on the next line.
            Object.defineProperty(this, "stack", { value: (new Error).stack, writable: true });
            this.stack = demangleAll(this.stack);
          }
        };
        FS.ErrnoError.prototype = new Error();
        FS.ErrnoError.prototype.constructor = FS.ErrnoError;
        // Some errors may happen quite a bit, to avoid overhead we reuse them (and suffer a lack of stack info)
        [44].forEach((code) => {
          FS.genericErrors[code] = new FS.ErrnoError(code);
          FS.genericErrors[code].stack = '<generic error, no stack>';
        });
      },staticInit:() => {
        FS.ensureErrnoError();
  
        FS.nameTable = new Array(4096);
  
        FS.mount(MEMFS, {}, '/');
  
        FS.createDefaultDirectories();
        FS.createDefaultDevices();
        FS.createSpecialDirectories();
  
        FS.filesystems = {
          'MEMFS': MEMFS,
          'IDBFS': IDBFS,
        };
      },init:(input, output, error) => {
        assert(!FS.init.initialized, 'FS.init was previously called. If you want to initialize later with custom parameters, remove any earlier calls (note that one is automatically added to the generated code)');
        FS.init.initialized = true;
  
        FS.ensureErrnoError();
  
        // Allow Module.stdin etc. to provide defaults, if none explicitly passed to us here
        Module['stdin'] = input || Module['stdin'];
        Module['stdout'] = output || Module['stdout'];
        Module['stderr'] = error || Module['stderr'];
  
        FS.createStandardStreams();
      },quit:() => {
        FS.init.initialized = false;
        // force-flush all streams, so we get musl std streams printed out
        _fflush(0);
        // close all of our streams
        for (var i = 0; i < FS.streams.length; i++) {
          var stream = FS.streams[i];
          if (!stream) {
            continue;
          }
          FS.close(stream);
        }
      },getMode:(canRead, canWrite) => {
        var mode = 0;
        if (canRead) mode |= 292 | 73;
        if (canWrite) mode |= 146;
        return mode;
      },findObject:(path, dontResolveLastLink) => {
        var ret = FS.analyzePath(path, dontResolveLastLink);
        if (!ret.exists) {
          return null;
        }
        return ret.object;
      },analyzePath:(path, dontResolveLastLink) => {
        // operate from within the context of the symlink's target
        try {
          var lookup = FS.lookupPath(path, { follow: !dontResolveLastLink });
          path = lookup.path;
        } catch (e) {
        }
        var ret = {
          isRoot: false, exists: false, error: 0, name: null, path: null, object: null,
          parentExists: false, parentPath: null, parentObject: null
        };
        try {
          var lookup = FS.lookupPath(path, { parent: true });
          ret.parentExists = true;
          ret.parentPath = lookup.path;
          ret.parentObject = lookup.node;
          ret.name = PATH.basename(path);
          lookup = FS.lookupPath(path, { follow: !dontResolveLastLink });
          ret.exists = true;
          ret.path = lookup.path;
          ret.object = lookup.node;
          ret.name = lookup.node.name;
          ret.isRoot = lookup.path === '/';
        } catch (e) {
          ret.error = e.errno;
        }        return ret;
      },createPath:(parent, path, canRead, canWrite) => {
        parent = typeof parent == 'string' ? parent : FS.getPath(parent);
        var parts = path.split('/').reverse();
        while (parts.length) {
          var part = parts.pop();
          if (!part) continue;
          var current = PATH.join2(parent, part);
          try {
            FS.mkdir(current);
          } catch (e) {
            // ignore EEXIST
          }
          parent = current;
        }
        return current;
      },createFile:(parent, name, properties, canRead, canWrite) => {
        var path = PATH.join2(typeof parent == 'string' ? parent : FS.getPath(parent), name);
        var mode = FS.getMode(canRead, canWrite);
        return FS.create(path, mode);
      },createDataFile:(parent, name, data, canRead, canWrite, canOwn) => {
        var path = name;
        if (parent) {
          parent = typeof parent == 'string' ? parent : FS.getPath(parent);
          path = name ? PATH.join2(parent, name) : parent;
        }
        var mode = FS.getMode(canRead, canWrite);
        var node = FS.create(path, mode);
        if (data) {
          if (typeof data == 'string') {
            var arr = new Array(data.length);
            for (var i = 0, len = data.length; i < len; ++i) arr[i] = data.charCodeAt(i);
            data = arr;
          }
          // make sure we can write to the file
          FS.chmod(node, mode | 146);
          var stream = FS.open(node, 577);
          FS.write(stream, data, 0, data.length, 0, canOwn);
          FS.close(stream);
          FS.chmod(node, mode);
        }
        return node;
      },createDevice:(parent, name, input, output) => {
        var path = PATH.join2(typeof parent == 'string' ? parent : FS.getPath(parent), name);
        var mode = FS.getMode(!!input, !!output);
        if (!FS.createDevice.major) FS.createDevice.major = 64;
        var dev = FS.makedev(FS.createDevice.major++, 0);
        // Create a fake device that a set of stream ops to emulate
        // the old behavior.
        FS.registerDevice(dev, {
          open: (stream) => {
            stream.seekable = false;
          },
          close: (stream) => {
            // flush any pending line data
            if (output && output.buffer && output.buffer.length) {
              output(10);
            }
          },
          read: (stream, buffer, offset, length, pos /* ignored */) => {
            var bytesRead = 0;
            for (var i = 0; i < length; i++) {
              var result;
              try {
                result = input();
              } catch (e) {
                throw new FS.ErrnoError(29);
              }
              if (result === undefined && bytesRead === 0) {
                throw new FS.ErrnoError(6);
              }
              if (result === null || result === undefined) break;
              bytesRead++;
              buffer[offset+i] = result;
            }
            if (bytesRead) {
              stream.node.timestamp = Date.now();
            }
            return bytesRead;
          },
          write: (stream, buffer, offset, length, pos) => {
            for (var i = 0; i < length; i++) {
              try {
                output(buffer[offset+i]);
              } catch (e) {
                throw new FS.ErrnoError(29);
              }
            }
            if (length) {
              stream.node.timestamp = Date.now();
            }
            return i;
          }
        });
        return FS.mkdev(path, mode, dev);
      },forceLoadFile:(obj) => {
        if (obj.isDevice || obj.isFolder || obj.link || obj.contents) return true;
        if (typeof XMLHttpRequest != 'undefined') {
          throw new Error("Lazy loading should have been performed (contents set) in createLazyFile, but it was not. Lazy loading only works in web workers. Use --embed-file or --preload-file in emcc on the main thread.");
        } else if (read_) {
          // Command-line.
          try {
            // WARNING: Can't read binary files in V8's d8 or tracemonkey's js, as
            //          read() will try to parse UTF8.
            obj.contents = intArrayFromString(read_(obj.url), true);
            obj.usedBytes = obj.contents.length;
          } catch (e) {
            throw new FS.ErrnoError(29);
          }
        } else {
          throw new Error('Cannot load without read() or XMLHttpRequest.');
        }
      },createLazyFile:(parent, name, url, canRead, canWrite) => {
        // Lazy chunked Uint8Array (implements get and length from Uint8Array). Actual getting is abstracted away for eventual reuse.
        /** @constructor */
        function LazyUint8Array() {
          this.lengthKnown = false;
          this.chunks = []; // Loaded chunks. Index is the chunk number
        }
        LazyUint8Array.prototype.get = /** @this{Object} */ function LazyUint8Array_get(idx) {
          if (idx > this.length-1 || idx < 0) {
            return undefined;
          }
          var chunkOffset = idx % this.chunkSize;
          var chunkNum = (idx / this.chunkSize)|0;
          return this.getter(chunkNum)[chunkOffset];
        };
        LazyUint8Array.prototype.setDataGetter = function LazyUint8Array_setDataGetter(getter) {
          this.getter = getter;
        };
        LazyUint8Array.prototype.cacheLength = function LazyUint8Array_cacheLength() {
          // Find length
          var xhr = new XMLHttpRequest();
          xhr.open('HEAD', url, false);
          xhr.send(null);
          if (!(xhr.status >= 200 && xhr.status < 300 || xhr.status === 304)) throw new Error("Couldn't load " + url + ". Status: " + xhr.status);
          var datalength = Number(xhr.getResponseHeader("Content-length"));
          var header;
          var hasByteServing = (header = xhr.getResponseHeader("Accept-Ranges")) && header === "bytes";
          var usesGzip = (header = xhr.getResponseHeader("Content-Encoding")) && header === "gzip";
  
          var chunkSize = 1024*1024; // Chunk size in bytes
  
          if (!hasByteServing) chunkSize = datalength;
  
          // Function to get a range from the remote URL.
          var doXHR = (from, to) => {
            if (from > to) throw new Error("invalid range (" + from + ", " + to + ") or no bytes requested!");
            if (to > datalength-1) throw new Error("only " + datalength + " bytes available! programmer error!");
  
            // TODO: Use mozResponseArrayBuffer, responseStream, etc. if available.
            var xhr = new XMLHttpRequest();
            xhr.open('GET', url, false);
            if (datalength !== chunkSize) xhr.setRequestHeader("Range", "bytes=" + from + "-" + to);
  
            // Some hints to the browser that we want binary data.
            xhr.responseType = 'arraybuffer';
            if (xhr.overrideMimeType) {
              xhr.overrideMimeType('text/plain; charset=x-user-defined');
            }
  
            xhr.send(null);
            if (!(xhr.status >= 200 && xhr.status < 300 || xhr.status === 304)) throw new Error("Couldn't load " + url + ". Status: " + xhr.status);
            if (xhr.response !== undefined) {
              return new Uint8Array(/** @type{Array<number>} */(xhr.response || []));
            }
            return intArrayFromString(xhr.responseText || '', true);
          };
          var lazyArray = this;
          lazyArray.setDataGetter((chunkNum) => {
            var start = chunkNum * chunkSize;
            var end = (chunkNum+1) * chunkSize - 1; // including this byte
            end = Math.min(end, datalength-1); // if datalength-1 is selected, this is the last block
            if (typeof lazyArray.chunks[chunkNum] == 'undefined') {
              lazyArray.chunks[chunkNum] = doXHR(start, end);
            }
            if (typeof lazyArray.chunks[chunkNum] == 'undefined') throw new Error('doXHR failed!');
            return lazyArray.chunks[chunkNum];
          });
  
          if (usesGzip || !datalength) {
            // if the server uses gzip or doesn't supply the length, we have to download the whole file to get the (uncompressed) length
            chunkSize = datalength = 1; // this will force getter(0)/doXHR do download the whole file
            datalength = this.getter(0).length;
            chunkSize = datalength;
            out("LazyFiles on gzip forces download of the whole file when length is accessed");
          }
  
          this._length = datalength;
          this._chunkSize = chunkSize;
          this.lengthKnown = true;
        };
        if (typeof XMLHttpRequest != 'undefined') {
          throw 'Cannot do synchronous binary XHRs outside webworkers in modern browsers. Use --embed-file or --preload-file in emcc';
          var lazyArray = new LazyUint8Array();
  
          var properties = { isDevice: false, contents: lazyArray };
        } else {
          var properties = { isDevice: false, url: url };
        }
  
        var node = FS.createFile(parent, name, properties, canRead, canWrite);
        // This is a total hack, but I want to get this lazy file code out of the
        // core of MEMFS. If we want to keep this lazy file concept I feel it should
        // be its own thin LAZYFS proxying calls to MEMFS.
        if (properties.contents) {
          node.contents = properties.contents;
        } else if (properties.url) {
          node.contents = null;
          node.url = properties.url;
        }
        // Add a function that defers querying the file size until it is asked the first time.
        Object.defineProperties(node, {
          usedBytes: {
            get: /** @this {FSNode} */ function() { return this.contents.length; }
          }
        });
        // override each stream op with one that tries to force load the lazy file first
        var stream_ops = {};
        var keys = Object.keys(node.stream_ops);
        keys.forEach((key) => {
          var fn = node.stream_ops[key];
          stream_ops[key] = function forceLoadLazyFile() {
            FS.forceLoadFile(node);
            return fn.apply(null, arguments);
          };
        });
        function writeChunks(stream, buffer, offset, length, position) {
          var contents = stream.node.contents;
          if (position >= contents.length)
            return 0;
          var size = Math.min(contents.length - position, length);
          assert(size >= 0);
          if (contents.slice) { // normal array
            for (var i = 0; i < size; i++) {
              buffer[offset + i] = contents[position + i];
            }
          } else {
            for (var i = 0; i < size; i++) { // LazyUint8Array from sync binary XHR
              buffer[offset + i] = contents.get(position + i);
            }
          }
          return size;
        }
        // use a custom read function
        stream_ops.read = (stream, buffer, offset, length, position) => {
          FS.forceLoadFile(node);
          return writeChunks(stream, buffer, offset, length, position)
        };
        // use a custom mmap function
        stream_ops.mmap = (stream, length, position, prot, flags) => {
          FS.forceLoadFile(node);
          var ptr = mmapAlloc();
          if (!ptr) {
            throw new FS.ErrnoError(48);
          }
          writeChunks(stream, HEAP8, ptr, length, position);
          return { ptr: ptr, allocated: true };
        };
        node.stream_ops = stream_ops;
        return node;
      },createPreloadedFile:(parent, name, url, canRead, canWrite, onload, onerror, dontCreateFile, canOwn, preFinish) => {
        // TODO we should allow people to just pass in a complete filename instead
        // of parent and name being that we just join them anyways
        var fullname = name ? PATH_FS.resolve(PATH.join2(parent, name)) : parent;
        var dep = getUniqueRunDependency('cp ' + fullname); // might have several active requests for the same fullname
        function processData(byteArray) {
          function finish(byteArray) {
            if (preFinish) preFinish();
            if (!dontCreateFile) {
              FS.createDataFile(parent, name, byteArray, canRead, canWrite, canOwn);
            }
            if (onload) onload();
            removeRunDependency(dep);
          }
          if (Browser.handledByPreloadPlugin(byteArray, fullname, finish, () => {
            if (onerror) onerror();
            removeRunDependency(dep);
          })) {
            return;
          }
          finish(byteArray);
        }
        addRunDependency(dep);
        if (typeof url == 'string') {
          asyncLoad(url, (byteArray) => processData(byteArray), onerror);
        } else {
          processData(url);
        }
      },absolutePath:() => {
        abort('FS.absolutePath has been removed; use PATH_FS.resolve instead');
      },createFolder:() => {
        abort('FS.createFolder has been removed; use FS.mkdir instead');
      },createLink:() => {
        abort('FS.createLink has been removed; use FS.symlink instead');
      },joinPath:() => {
        abort('FS.joinPath has been removed; use PATH.join instead');
      },mmapAlloc:() => {
        abort('FS.mmapAlloc has been replaced by the top level function mmapAlloc');
      },standardizePath:() => {
        abort('FS.standardizePath has been removed; use PATH.normalize instead');
      }};
  
  var SYSCALLS = {DEFAULT_POLLMASK:5,calculateAt:function(dirfd, path, allowEmpty) {
        if (PATH.isAbs(path)) {
          return path;
        }
        // relative path
        var dir;
        if (dirfd === -100) {
          dir = FS.cwd();
        } else {
          var dirstream = SYSCALLS.getStreamFromFD(dirfd);
          dir = dirstream.path;
        }
        if (path.length == 0) {
          if (!allowEmpty) {
            throw new FS.ErrnoError(44);          }
          return dir;
        }
        return PATH.join2(dir, path);
      },doStat:function(func, path, buf) {
        try {
          var stat = func(path);
        } catch (e) {
          if (e && e.node && PATH.normalize(path) !== PATH.normalize(FS.getPath(e.node))) {
            // an error occurred while trying to look up the path; we should just report ENOTDIR
            return -54;
          }
          throw e;
        }
        HEAP32[((buf)>>2)] = stat.dev;
        HEAP32[(((buf)+(8))>>2)] = stat.ino;
        HEAP32[(((buf)+(12))>>2)] = stat.mode;
        HEAPU32[(((buf)+(16))>>2)] = stat.nlink;
        HEAP32[(((buf)+(20))>>2)] = stat.uid;
        HEAP32[(((buf)+(24))>>2)] = stat.gid;
        HEAP32[(((buf)+(28))>>2)] = stat.rdev;
        (tempI64 = [stat.size>>>0,(tempDouble=stat.size,(+(Math.abs(tempDouble))) >= 1.0 ? (tempDouble > 0.0 ? ((Math.min((+(Math.floor((tempDouble)/4294967296.0))), 4294967295.0))|0)>>>0 : (~~((+(Math.ceil((tempDouble - +(((~~(tempDouble)))>>>0))/4294967296.0)))))>>>0) : 0)],HEAP32[(((buf)+(40))>>2)] = tempI64[0],HEAP32[(((buf)+(44))>>2)] = tempI64[1]);
        HEAP32[(((buf)+(48))>>2)] = 4096;
        HEAP32[(((buf)+(52))>>2)] = stat.blocks;
        var atime = stat.atime.getTime();
        var mtime = stat.mtime.getTime();
        var ctime = stat.ctime.getTime();
        (tempI64 = [Math.floor(atime / 1000)>>>0,(tempDouble=Math.floor(atime / 1000),(+(Math.abs(tempDouble))) >= 1.0 ? (tempDouble > 0.0 ? ((Math.min((+(Math.floor((tempDouble)/4294967296.0))), 4294967295.0))|0)>>>0 : (~~((+(Math.ceil((tempDouble - +(((~~(tempDouble)))>>>0))/4294967296.0)))))>>>0) : 0)],HEAP32[(((buf)+(56))>>2)] = tempI64[0],HEAP32[(((buf)+(60))>>2)] = tempI64[1]);
        HEAPU32[(((buf)+(64))>>2)] = (atime % 1000) * 1000;
        (tempI64 = [Math.floor(mtime / 1000)>>>0,(tempDouble=Math.floor(mtime / 1000),(+(Math.abs(tempDouble))) >= 1.0 ? (tempDouble > 0.0 ? ((Math.min((+(Math.floor((tempDouble)/4294967296.0))), 4294967295.0))|0)>>>0 : (~~((+(Math.ceil((tempDouble - +(((~~(tempDouble)))>>>0))/4294967296.0)))))>>>0) : 0)],HEAP32[(((buf)+(72))>>2)] = tempI64[0],HEAP32[(((buf)+(76))>>2)] = tempI64[1]);
        HEAPU32[(((buf)+(80))>>2)] = (mtime % 1000) * 1000;
        (tempI64 = [Math.floor(ctime / 1000)>>>0,(tempDouble=Math.floor(ctime / 1000),(+(Math.abs(tempDouble))) >= 1.0 ? (tempDouble > 0.0 ? ((Math.min((+(Math.floor((tempDouble)/4294967296.0))), 4294967295.0))|0)>>>0 : (~~((+(Math.ceil((tempDouble - +(((~~(tempDouble)))>>>0))/4294967296.0)))))>>>0) : 0)],HEAP32[(((buf)+(88))>>2)] = tempI64[0],HEAP32[(((buf)+(92))>>2)] = tempI64[1]);
        HEAPU32[(((buf)+(96))>>2)] = (ctime % 1000) * 1000;
        (tempI64 = [stat.ino>>>0,(tempDouble=stat.ino,(+(Math.abs(tempDouble))) >= 1.0 ? (tempDouble > 0.0 ? ((Math.min((+(Math.floor((tempDouble)/4294967296.0))), 4294967295.0))|0)>>>0 : (~~((+(Math.ceil((tempDouble - +(((~~(tempDouble)))>>>0))/4294967296.0)))))>>>0) : 0)],HEAP32[(((buf)+(104))>>2)] = tempI64[0],HEAP32[(((buf)+(108))>>2)] = tempI64[1]);
        return 0;
      },doMsync:function(addr, stream, len, flags, offset) {
        if (!FS.isFile(stream.node.mode)) {
          throw new FS.ErrnoError(43);
        }
        if (flags & 2) {
          // MAP_PRIVATE calls need not to be synced back to underlying fs
          return 0;
        }
        var buffer = HEAPU8.slice(addr, addr + len);
        FS.msync(stream, buffer, offset, len, flags);
      },varargs:undefined,get:function() {
        assert(SYSCALLS.varargs != undefined);
        SYSCALLS.varargs += 4;
        var ret = HEAP32[(((SYSCALLS.varargs)-(4))>>2)];
        return ret;
      },getStr:function(ptr) {
        var ret = UTF8ToString(ptr);
        return ret;
      },getStreamFromFD:function(fd) {
        var stream = FS.getStream(fd);
        if (!stream) throw new FS.ErrnoError(8);
        return stream;
      }};
  function ___syscall_fcntl64(fd, cmd, varargs) {
  SYSCALLS.varargs = varargs;
  try {
  
      var stream = SYSCALLS.getStreamFromFD(fd);
      switch (cmd) {
        case 0: {
          var arg = SYSCALLS.get();
          if (arg < 0) {
            return -28;
          }
          var newStream;
          newStream = FS.createStream(stream, arg);
          return newStream.fd;
        }
        case 1:
        case 2:
          return 0;  // FD_CLOEXEC makes no sense for a single process.
        case 3:
          return stream.flags;
        case 4: {
          var arg = SYSCALLS.get();
          stream.flags |= arg;
          return 0;
        }
        case 5:
        /* case 5: Currently in musl F_GETLK64 has same value as F_GETLK, so omitted to avoid duplicate case blocks. If that changes, uncomment this */ {
          
          var arg = SYSCALLS.get();
          var offset = 0;
          // We're always unlocked.
          HEAP16[(((arg)+(offset))>>1)] = 2;
          return 0;
        }
        case 6:
        case 7:
        /* case 6: Currently in musl F_SETLK64 has same value as F_SETLK, so omitted to avoid duplicate case blocks. If that changes, uncomment this */
        /* case 7: Currently in musl F_SETLKW64 has same value as F_SETLKW, so omitted to avoid duplicate case blocks. If that changes, uncomment this */
          
          
          return 0; // Pretend that the locking is successful.
        case 16:
        case 8:
          return -28; // These are for sockets. We don't have them fully implemented yet.
        case 9:
          // musl trusts getown return values, due to a bug where they must be, as they overlap with errors. just return -1 here, so fcntl() returns that, and we set errno ourselves.
          setErrNo(28);
          return -1;
        default: {
          return -28;
        }
      }
    } catch (e) {
    if (typeof FS == 'undefined' || !(e.name === 'ErrnoError')) throw e;
    return -e.errno;
  }
  }

  function ___syscall_fstat64(fd, buf) {
  try {
  
      var stream = SYSCALLS.getStreamFromFD(fd);
      return SYSCALLS.doStat(FS.stat, stream.path, buf);
    } catch (e) {
    if (typeof FS == 'undefined' || !(e.name === 'ErrnoError')) throw e;
    return -e.errno;
  }
  }

  function ___syscall_ioctl(fd, op, varargs) {
  SYSCALLS.varargs = varargs;
  try {
  
      var stream = SYSCALLS.getStreamFromFD(fd);
      switch (op) {
        case 21509:
        case 21505: {
          if (!stream.tty) return -59;
          return 0;
        }
        case 21510:
        case 21511:
        case 21512:
        case 21506:
        case 21507:
        case 21508: {
          if (!stream.tty) return -59;
          return 0; // no-op, not actually adjusting terminal settings
        }
        case 21519: {
          if (!stream.tty) return -59;
          var argp = SYSCALLS.get();
          HEAP32[((argp)>>2)] = 0;
          return 0;
        }
        case 21520: {
          if (!stream.tty) return -59;
          return -28; // not supported
        }
        case 21531: {
          var argp = SYSCALLS.get();
          return FS.ioctl(stream, op, argp);
        }
        case 21523: {
          // TODO: in theory we should write to the winsize struct that gets
          // passed in, but for now musl doesn't read anything on it
          if (!stream.tty) return -59;
          return 0;
        }
        case 21524: {
          // TODO: technically, this ioctl call should change the window size.
          // but, since emscripten doesn't have any concept of a terminal window
          // yet, we'll just silently throw it away as we do TIOCGWINSZ
          if (!stream.tty) return -59;
          return 0;
        }
        default: return -28; // not supported
      }
    } catch (e) {
    if (typeof FS == 'undefined' || !(e.name === 'ErrnoError')) throw e;
    return -e.errno;
  }
  }

  function ___syscall_lstat64(path, buf) {
  try {
  
      path = SYSCALLS.getStr(path);
      return SYSCALLS.doStat(FS.lstat, path, buf);
    } catch (e) {
    if (typeof FS == 'undefined' || !(e.name === 'ErrnoError')) throw e;
    return -e.errno;
  }
  }

  function ___syscall_newfstatat(dirfd, path, buf, flags) {
  try {
  
      path = SYSCALLS.getStr(path);
      var nofollow = flags & 256;
      var allowEmpty = flags & 4096;
      flags = flags & (~6400);
      assert(!flags, 'unknown flags in __syscall_newfstatat: ' + flags);
      path = SYSCALLS.calculateAt(dirfd, path, allowEmpty);
      return SYSCALLS.doStat(nofollow ? FS.lstat : FS.stat, path, buf);
    } catch (e) {
    if (typeof FS == 'undefined' || !(e.name === 'ErrnoError')) throw e;
    return -e.errno;
  }
  }

  function ___syscall_openat(dirfd, path, flags, varargs) {
  SYSCALLS.varargs = varargs;
  try {
  
      path = SYSCALLS.getStr(path);
      path = SYSCALLS.calculateAt(dirfd, path);
      var mode = varargs ? SYSCALLS.get() : 0;
      return FS.open(path, flags, mode).fd;
    } catch (e) {
    if (typeof FS == 'undefined' || !(e.name === 'ErrnoError')) throw e;
    return -e.errno;
  }
  }

  function ___syscall_stat64(path, buf) {
  try {
  
      path = SYSCALLS.getStr(path);
      return SYSCALLS.doStat(FS.stat, path, buf);
    } catch (e) {
    if (typeof FS == 'undefined' || !(e.name === 'ErrnoError')) throw e;
    return -e.errno;
  }
  }

  
  function ___throw_exception_with_stack_trace(ex) {
      var e = new WebAssembly.Exception(getCppExceptionTag(), [ex], {traceStack: true});
      e.message = getExceptionMessage(e);
      // The generated stack trace will be in the form of:
      //
      // Error
      //     at ___throw_exception_with_stack_trace(test.js:1139:13)
      //     at __cxa_throw (wasm://wasm/009a7c9a:wasm-function[1551]:0x24367)
      //     ...
      //
      // Remove this JS function name, which is in the second line, from the stack
      // trace. Note that .stack does not yet exist in all browsers (see #18828).
      if (e.stack) {
        var arr = e.stack.split('\n');
        arr.splice(1,1);
        e.stack = arr.join('\n');
      }
      throw e;
    }

  function __embind_register_bigint(primitiveType, name, size, minRange, maxRange) {}

  function getShiftFromSize(size) {
      switch (size) {
          case 1: return 0;
          case 2: return 1;
          case 4: return 2;
          case 8: return 3;
          default:
              throw new TypeError('Unknown type size: ' + size);
      }
    }
  
  function embind_init_charCodes() {
      var codes = new Array(256);
      for (var i = 0; i < 256; ++i) {
          codes[i] = String.fromCharCode(i);
      }
      embind_charCodes = codes;
    }
  var embind_charCodes = undefined;
  function readLatin1String(ptr) {
      var ret = "";
      var c = ptr;
      while (HEAPU8[c]) {
          ret += embind_charCodes[HEAPU8[c++]];
      }
      return ret;
    }
  
  var awaitingDependencies = {};
  
  var registeredTypes = {};
  
  var typeDependencies = {};
  
  var char_0 = 48;
  
  var char_9 = 57;
  function makeLegalFunctionName(name) {
      if (undefined === name) {
        return '_unknown';
      }
      name = name.replace(/[^a-zA-Z0-9_]/g, '$');
      var f = name.charCodeAt(0);
      if (f >= char_0 && f <= char_9) {
        return '_' + name;
      }
      return name;
    }
  function createNamedFunction(name, body) {
      name = makeLegalFunctionName(name);
      // Use an abject with a computed property name to create a new function with
      // a name specified at runtime, but without using `new Function` or `eval`.
      return {
        [name]: function() {
          return body.apply(this, arguments);
        }
      }[name];
    }
  function extendError(baseErrorType, errorName) {
      var errorClass = createNamedFunction(errorName, function(message) {
        this.name = errorName;
        this.message = message;
  
        var stack = (new Error(message)).stack;
        if (stack !== undefined) {
          this.stack = this.toString() + '\n' +
              stack.replace(/^Error(:[^\n]*)?\n/, '');
        }
      });
      errorClass.prototype = Object.create(baseErrorType.prototype);
      errorClass.prototype.constructor = errorClass;
      errorClass.prototype.toString = function() {
        if (this.message === undefined) {
          return this.name;
        } else {
          return this.name + ': ' + this.message;
        }
      };
  
      return errorClass;
    }
  var BindingError = undefined;
  function throwBindingError(message) {
      throw new BindingError(message);
    }
  
  
  
  
  var InternalError = undefined;
  function throwInternalError(message) {
      throw new InternalError(message);
    }
  function whenDependentTypesAreResolved(myTypes, dependentTypes, getTypeConverters) {
      myTypes.forEach(function(type) {
          typeDependencies[type] = dependentTypes;
      });
  
      function onComplete(typeConverters) {
          var myTypeConverters = getTypeConverters(typeConverters);
          if (myTypeConverters.length !== myTypes.length) {
              throwInternalError('Mismatched type converter count');
          }
          for (var i = 0; i < myTypes.length; ++i) {
              registerType(myTypes[i], myTypeConverters[i]);
          }
      }
  
      var typeConverters = new Array(dependentTypes.length);
      var unregisteredTypes = [];
      var registered = 0;
      dependentTypes.forEach((dt, i) => {
        if (registeredTypes.hasOwnProperty(dt)) {
          typeConverters[i] = registeredTypes[dt];
        } else {
          unregisteredTypes.push(dt);
          if (!awaitingDependencies.hasOwnProperty(dt)) {
            awaitingDependencies[dt] = [];
          }
          awaitingDependencies[dt].push(() => {
            typeConverters[i] = registeredTypes[dt];
            ++registered;
            if (registered === unregisteredTypes.length) {
              onComplete(typeConverters);
            }
          });
        }
      });
      if (0 === unregisteredTypes.length) {
        onComplete(typeConverters);
      }
    }
  /** @param {Object=} options */
  function registerType(rawType, registeredInstance, options = {}) {
      if (!('argPackAdvance' in registeredInstance)) {
          throw new TypeError('registerType registeredInstance requires argPackAdvance');
      }
  
      var name = registeredInstance.name;
      if (!rawType) {
          throwBindingError('type "' + name + '" must have a positive integer typeid pointer');
      }
      if (registeredTypes.hasOwnProperty(rawType)) {
          if (options.ignoreDuplicateRegistrations) {
              return;
          } else {
              throwBindingError("Cannot register type '" + name + "' twice");
          }
      }
  
      registeredTypes[rawType] = registeredInstance;
      delete typeDependencies[rawType];
  
      if (awaitingDependencies.hasOwnProperty(rawType)) {
        var callbacks = awaitingDependencies[rawType];
        delete awaitingDependencies[rawType];
        callbacks.forEach((cb) => cb());
      }
    }
  function __embind_register_bool(rawType, name, size, trueValue, falseValue) {
      var shift = getShiftFromSize(size);
  
      name = readLatin1String(name);
      registerType(rawType, {
          name: name,
          'fromWireType': function(wt) {
              // ambiguous emscripten ABI: sometimes return values are
              // true or false, and sometimes integers (0 or 1)
              return !!wt;
          },
          'toWireType': function(destructors, o) {
              return o ? trueValue : falseValue;
          },
          'argPackAdvance': 8,
          'readValueFromPointer': function(pointer) {
              // TODO: if heap is fixed (like in asm.js) this could be executed outside
              var heap;
              if (size === 1) {
                  heap = HEAP8;
              } else if (size === 2) {
                  heap = HEAP16;
              } else if (size === 4) {
                  heap = HEAP32;
              } else {
                  throw new TypeError("Unknown boolean type size: " + name);
              }
              return this['fromWireType'](heap[pointer >> shift]);
          },
          destructorFunction: null, // This type does not need a destructor
      });
    }

  
  function ClassHandle_isAliasOf(other) {
      if (!(this instanceof ClassHandle)) {
        return false;
      }
      if (!(other instanceof ClassHandle)) {
        return false;
      }
  
      var leftClass = this.$$.ptrType.registeredClass;
      var left = this.$$.ptr;
      var rightClass = other.$$.ptrType.registeredClass;
      var right = other.$$.ptr;
  
      while (leftClass.baseClass) {
        left = leftClass.upcast(left);
        leftClass = leftClass.baseClass;
      }
  
      while (rightClass.baseClass) {
        right = rightClass.upcast(right);
        rightClass = rightClass.baseClass;
      }
  
      return leftClass === rightClass && left === right;
    }
  
  function shallowCopyInternalPointer(o) {
      return {
        count: o.count,
        deleteScheduled: o.deleteScheduled,
        preservePointerOnDelete: o.preservePointerOnDelete,
        ptr: o.ptr,
        ptrType: o.ptrType,
        smartPtr: o.smartPtr,
        smartPtrType: o.smartPtrType,
      };
    }
  
  function throwInstanceAlreadyDeleted(obj) {
      function getInstanceTypeName(handle) {
        return handle.$$.ptrType.registeredClass.name;
      }
      throwBindingError(getInstanceTypeName(obj) + ' instance already deleted');
    }
  
  var finalizationRegistry = false;
  
  function detachFinalizer(handle) {}
  
  function runDestructor($$) {
      if ($$.smartPtr) {
        $$.smartPtrType.rawDestructor($$.smartPtr);
      } else {
        $$.ptrType.registeredClass.rawDestructor($$.ptr);
      }
    }
  function releaseClassHandle($$) {
      $$.count.value -= 1;
      var toDelete = 0 === $$.count.value;
      if (toDelete) {
        runDestructor($$);
      }
    }
  
  function downcastPointer(ptr, ptrClass, desiredClass) {
      if (ptrClass === desiredClass) {
        return ptr;
      }
      if (undefined === desiredClass.baseClass) {
        return null; // no conversion
      }
  
      var rv = downcastPointer(ptr, ptrClass, desiredClass.baseClass);
      if (rv === null) {
        return null;
      }
      return desiredClass.downcast(rv);
    }
  
  var registeredPointers = {};
  
  function getInheritedInstanceCount() {
      return Object.keys(registeredInstances).length;
    }
  
  function getLiveInheritedInstances() {
      var rv = [];
      for (var k in registeredInstances) {
        if (registeredInstances.hasOwnProperty(k)) {
          rv.push(registeredInstances[k]);
        }
      }
      return rv;
    }
  
  var deletionQueue = [];
  function flushPendingDeletes() {
      while (deletionQueue.length) {
        var obj = deletionQueue.pop();
        obj.$$.deleteScheduled = false;
        obj['delete']();
      }
    }
  
  var delayFunction = undefined;
  
  
  function setDelayFunction(fn) {
      delayFunction = fn;
      if (deletionQueue.length && delayFunction) {
        delayFunction(flushPendingDeletes);
      }
    }
  function init_embind() {
      Module['getInheritedInstanceCount'] = getInheritedInstanceCount;
      Module['getLiveInheritedInstances'] = getLiveInheritedInstances;
      Module['flushPendingDeletes'] = flushPendingDeletes;
      Module['setDelayFunction'] = setDelayFunction;
    }
  var registeredInstances = {};
  
  function getBasestPointer(class_, ptr) {
      if (ptr === undefined) {
          throwBindingError('ptr should not be undefined');
      }
      while (class_.baseClass) {
          ptr = class_.upcast(ptr);
          class_ = class_.baseClass;
      }
      return ptr;
    }
  function getInheritedInstance(class_, ptr) {
      ptr = getBasestPointer(class_, ptr);
      return registeredInstances[ptr];
    }
  
  
  function makeClassHandle(prototype, record) {
      if (!record.ptrType || !record.ptr) {
        throwInternalError('makeClassHandle requires ptr and ptrType');
      }
      var hasSmartPtrType = !!record.smartPtrType;
      var hasSmartPtr = !!record.smartPtr;
      if (hasSmartPtrType !== hasSmartPtr) {
        throwInternalError('Both smartPtrType and smartPtr must be specified');
      }
      record.count = { value: 1 };
      return attachFinalizer(Object.create(prototype, {
        $$: {
            value: record,
        },
      }));
    }
  function RegisteredPointer_fromWireType(ptr) {
      // ptr is a raw pointer (or a raw smartpointer)
  
      // rawPointer is a maybe-null raw pointer
      var rawPointer = this.getPointee(ptr);
      if (!rawPointer) {
        this.destructor(ptr);
        return null;
      }
  
      var registeredInstance = getInheritedInstance(this.registeredClass, rawPointer);
      if (undefined !== registeredInstance) {
        // JS object has been neutered, time to repopulate it
        if (0 === registeredInstance.$$.count.value) {
          registeredInstance.$$.ptr = rawPointer;
          registeredInstance.$$.smartPtr = ptr;
          return registeredInstance['clone']();
        } else {
          // else, just increment reference count on existing object
          // it already has a reference to the smart pointer
          var rv = registeredInstance['clone']();
          this.destructor(ptr);
          return rv;
        }
      }
  
      function makeDefaultHandle() {
        if (this.isSmartPointer) {
          return makeClassHandle(this.registeredClass.instancePrototype, {
            ptrType: this.pointeeType,
            ptr: rawPointer,
            smartPtrType: this,
            smartPtr: ptr,
          });
        } else {
          return makeClassHandle(this.registeredClass.instancePrototype, {
            ptrType: this,
            ptr: ptr,
          });
        }
      }
  
      var actualType = this.registeredClass.getActualType(rawPointer);
      var registeredPointerRecord = registeredPointers[actualType];
      if (!registeredPointerRecord) {
        return makeDefaultHandle.call(this);
      }
  
      var toType;
      if (this.isConst) {
        toType = registeredPointerRecord.constPointerType;
      } else {
        toType = registeredPointerRecord.pointerType;
      }
      var dp = downcastPointer(
          rawPointer,
          this.registeredClass,
          toType.registeredClass);
      if (dp === null) {
        return makeDefaultHandle.call(this);
      }
      if (this.isSmartPointer) {
        return makeClassHandle(toType.registeredClass.instancePrototype, {
          ptrType: toType,
          ptr: dp,
          smartPtrType: this,
          smartPtr: ptr,
        });
      } else {
        return makeClassHandle(toType.registeredClass.instancePrototype, {
          ptrType: toType,
          ptr: dp,
        });
      }
    }
  function attachFinalizer(handle) {
      if ('undefined' === typeof FinalizationRegistry) {
        attachFinalizer = (handle) => handle;
        return handle;
      }
      // If the running environment has a FinalizationRegistry (see
      // https://github.com/tc39/proposal-weakrefs), then attach finalizers
      // for class handles.  We check for the presence of FinalizationRegistry
      // at run-time, not build-time.
      finalizationRegistry = new FinalizationRegistry((info) => {
        console.warn(info.leakWarning.stack.replace(/^Error: /, ''));
        releaseClassHandle(info.$$);
      });
      attachFinalizer = (handle) => {
        var $$ = handle.$$;
        var hasSmartPtr = !!$$.smartPtr;
        if (hasSmartPtr) {
          // We should not call the destructor on raw pointers in case other code expects the pointee to live
          var info = { $$: $$ };
          // Create a warning as an Error instance in advance so that we can store
          // the current stacktrace and point to it when / if a leak is detected.
          // This is more useful than the empty stacktrace of `FinalizationRegistry`
          // callback.
          var cls = $$.ptrType.registeredClass;
          info.leakWarning = new Error("Embind found a leaked C++ instance " + cls.name + " <" + ptrToString($$.ptr) + ">.\n" +
          "We'll free it automatically in this case, but this functionality is not reliable across various environments.\n" +
          "Make sure to invoke .delete() manually once you're done with the instance instead.\n" +
          "Originally allocated"); // `.stack` will add "at ..." after this sentence
          if ('captureStackTrace' in Error) {
            Error.captureStackTrace(info.leakWarning, RegisteredPointer_fromWireType);
          }
          finalizationRegistry.register(handle, info, handle);
        }
        return handle;
      };
      detachFinalizer = (handle) => finalizationRegistry.unregister(handle);
      return attachFinalizer(handle);
    }
  function ClassHandle_clone() {
      if (!this.$$.ptr) {
        throwInstanceAlreadyDeleted(this);
      }
  
      if (this.$$.preservePointerOnDelete) {
        this.$$.count.value += 1;
        return this;
      } else {
        var clone = attachFinalizer(Object.create(Object.getPrototypeOf(this), {
          $$: {
            value: shallowCopyInternalPointer(this.$$),
          }
        }));
  
        clone.$$.count.value += 1;
        clone.$$.deleteScheduled = false;
        return clone;
      }
    }
  
  
  
  
  function ClassHandle_delete() {
      if (!this.$$.ptr) {
        throwInstanceAlreadyDeleted(this);
      }
  
      if (this.$$.deleteScheduled && !this.$$.preservePointerOnDelete) {
        throwBindingError('Object already scheduled for deletion');
      }
  
      detachFinalizer(this);
      releaseClassHandle(this.$$);
  
      if (!this.$$.preservePointerOnDelete) {
        this.$$.smartPtr = undefined;
        this.$$.ptr = undefined;
      }
    }
  
  function ClassHandle_isDeleted() {
      return !this.$$.ptr;
    }
  
  
  
  function ClassHandle_deleteLater() {
      if (!this.$$.ptr) {
        throwInstanceAlreadyDeleted(this);
      }
      if (this.$$.deleteScheduled && !this.$$.preservePointerOnDelete) {
        throwBindingError('Object already scheduled for deletion');
      }
      deletionQueue.push(this);
      if (deletionQueue.length === 1 && delayFunction) {
        delayFunction(flushPendingDeletes);
      }
      this.$$.deleteScheduled = true;
      return this;
    }
  function init_ClassHandle() {
      ClassHandle.prototype['isAliasOf'] = ClassHandle_isAliasOf;
      ClassHandle.prototype['clone'] = ClassHandle_clone;
      ClassHandle.prototype['delete'] = ClassHandle_delete;
      ClassHandle.prototype['isDeleted'] = ClassHandle_isDeleted;
      ClassHandle.prototype['deleteLater'] = ClassHandle_deleteLater;
    }
  function ClassHandle() {
    }
  
  
  
  function ensureOverloadTable(proto, methodName, humanName) {
      if (undefined === proto[methodName].overloadTable) {
        var prevFunc = proto[methodName];
        // Inject an overload resolver function that routes to the appropriate overload based on the number of arguments.
        proto[methodName] = function() {
          // TODO This check can be removed in -O3 level "unsafe" optimizations.
          if (!proto[methodName].overloadTable.hasOwnProperty(arguments.length)) {
              throwBindingError("Function '" + humanName + "' called with an invalid number of arguments (" + arguments.length + ") - expects one of (" + proto[methodName].overloadTable + ")!");
          }
          return proto[methodName].overloadTable[arguments.length].apply(this, arguments);
        };
        // Move the previous function into the overload table.
        proto[methodName].overloadTable = [];
        proto[methodName].overloadTable[prevFunc.argCount] = prevFunc;
      }
    }
  
  /** @param {number=} numArguments */
  function exposePublicSymbol(name, value, numArguments) {
      if (Module.hasOwnProperty(name)) {
        if (undefined === numArguments || (undefined !== Module[name].overloadTable && undefined !== Module[name].overloadTable[numArguments])) {
          throwBindingError("Cannot register public name '" + name + "' twice");
        }
  
        // We are exposing a function with the same name as an existing function. Create an overload table and a function selector
        // that routes between the two.
        ensureOverloadTable(Module, name, name);
        if (Module.hasOwnProperty(numArguments)) {
            throwBindingError("Cannot register multiple overloads of a function with the same number of arguments (" + numArguments + ")!");
        }
        // Add the new function into the overload table.
        Module[name].overloadTable[numArguments] = value;
      }
      else {
        Module[name] = value;
        if (undefined !== numArguments) {
          Module[name].numArguments = numArguments;
        }
      }
    }
  
  
  
  /** @constructor */
  function RegisteredClass(name,
                               constructor,
                               instancePrototype,
                               rawDestructor,
                               baseClass,
                               getActualType,
                               upcast,
                               downcast) {
      this.name = name;
      this.constructor = constructor;
      this.instancePrototype = instancePrototype;
      this.rawDestructor = rawDestructor;
      this.baseClass = baseClass;
      this.getActualType = getActualType;
      this.upcast = upcast;
      this.downcast = downcast;
      this.pureVirtualFunctions = [];
    }
  
  
  function upcastPointer(ptr, ptrClass, desiredClass) {
      while (ptrClass !== desiredClass) {
        if (!ptrClass.upcast) {
          throwBindingError("Expected null or instance of " + desiredClass.name + ", got an instance of " + ptrClass.name);
        }
        ptr = ptrClass.upcast(ptr);
        ptrClass = ptrClass.baseClass;
      }
      return ptr;
    }
  function constNoSmartPtrRawPointerToWireType(destructors, handle) {
      if (handle === null) {
        if (this.isReference) {
          throwBindingError('null is not a valid ' + this.name);
        }
        return 0;
      }
  
      if (!handle.$$) {
        throwBindingError('Cannot pass "' + embindRepr(handle) + '" as a ' + this.name);
      }
      if (!handle.$$.ptr) {
        throwBindingError('Cannot pass deleted object as a pointer of type ' + this.name);
      }
      var handleClass = handle.$$.ptrType.registeredClass;
      var ptr = upcastPointer(handle.$$.ptr, handleClass, this.registeredClass);
      return ptr;
    }
  
  
  function genericPointerToWireType(destructors, handle) {
      var ptr;
      if (handle === null) {
        if (this.isReference) {
          throwBindingError('null is not a valid ' + this.name);
        }
  
        if (this.isSmartPointer) {
          ptr = this.rawConstructor();
          if (destructors !== null) {
            destructors.push(this.rawDestructor, ptr);
          }
          return ptr;
        } else {
          return 0;
        }
      }
  
      if (!handle.$$) {
        throwBindingError('Cannot pass "' + embindRepr(handle) + '" as a ' + this.name);
      }
      if (!handle.$$.ptr) {
        throwBindingError('Cannot pass deleted object as a pointer of type ' + this.name);
      }
      if (!this.isConst && handle.$$.ptrType.isConst) {
        throwBindingError('Cannot convert argument of type ' + (handle.$$.smartPtrType ? handle.$$.smartPtrType.name : handle.$$.ptrType.name) + ' to parameter type ' + this.name);
      }
      var handleClass = handle.$$.ptrType.registeredClass;
      ptr = upcastPointer(handle.$$.ptr, handleClass, this.registeredClass);
  
      if (this.isSmartPointer) {
        // TODO: this is not strictly true
        // We could support BY_EMVAL conversions from raw pointers to smart pointers
        // because the smart pointer can hold a reference to the handle
        if (undefined === handle.$$.smartPtr) {
          throwBindingError('Passing raw pointer to smart pointer is illegal');
        }
  
        switch (this.sharingPolicy) {
          case 0: // NONE
            // no upcasting
            if (handle.$$.smartPtrType === this) {
              ptr = handle.$$.smartPtr;
            } else {
              throwBindingError('Cannot convert argument of type ' + (handle.$$.smartPtrType ? handle.$$.smartPtrType.name : handle.$$.ptrType.name) + ' to parameter type ' + this.name);
            }
            break;
  
          case 1: // INTRUSIVE
            ptr = handle.$$.smartPtr;
            break;
  
          case 2: // BY_EMVAL
            if (handle.$$.smartPtrType === this) {
              ptr = handle.$$.smartPtr;
            } else {
              var clonedHandle = handle['clone']();
              ptr = this.rawShare(
                ptr,
                Emval.toHandle(function() {
                  clonedHandle['delete']();
                })
              );
              if (destructors !== null) {
                destructors.push(this.rawDestructor, ptr);
              }
            }
            break;
  
          default:
            throwBindingError('Unsupporting sharing policy');
        }
      }
      return ptr;
    }
  
  
  function nonConstNoSmartPtrRawPointerToWireType(destructors, handle) {
      if (handle === null) {
        if (this.isReference) {
          throwBindingError('null is not a valid ' + this.name);
        }
        return 0;
      }
  
      if (!handle.$$) {
        throwBindingError('Cannot pass "' + embindRepr(handle) + '" as a ' + this.name);
      }
      if (!handle.$$.ptr) {
        throwBindingError('Cannot pass deleted object as a pointer of type ' + this.name);
      }
      if (handle.$$.ptrType.isConst) {
          throwBindingError('Cannot convert argument of type ' + handle.$$.ptrType.name + ' to parameter type ' + this.name);
      }
      var handleClass = handle.$$.ptrType.registeredClass;
      var ptr = upcastPointer(handle.$$.ptr, handleClass, this.registeredClass);
      return ptr;
    }
  
  function simpleReadValueFromPointer(pointer) {
      return this['fromWireType'](HEAP32[((pointer)>>2)]);
    }
  
  function RegisteredPointer_getPointee(ptr) {
      if (this.rawGetPointee) {
        ptr = this.rawGetPointee(ptr);
      }
      return ptr;
    }
  
  function RegisteredPointer_destructor(ptr) {
      if (this.rawDestructor) {
        this.rawDestructor(ptr);
      }
    }
  
  function RegisteredPointer_deleteObject(handle) {
      if (handle !== null) {
        handle['delete']();
      }
    }
  
  function init_RegisteredPointer() {
      RegisteredPointer.prototype.getPointee = RegisteredPointer_getPointee;
      RegisteredPointer.prototype.destructor = RegisteredPointer_destructor;
      RegisteredPointer.prototype['argPackAdvance'] = 8;
      RegisteredPointer.prototype['readValueFromPointer'] = simpleReadValueFromPointer;
      RegisteredPointer.prototype['deleteObject'] = RegisteredPointer_deleteObject;
      RegisteredPointer.prototype['fromWireType'] = RegisteredPointer_fromWireType;
    }
  /** @constructor
      @param {*=} pointeeType,
      @param {*=} sharingPolicy,
      @param {*=} rawGetPointee,
      @param {*=} rawConstructor,
      @param {*=} rawShare,
      @param {*=} rawDestructor,
       */
  function RegisteredPointer(
      name,
      registeredClass,
      isReference,
      isConst,
  
      // smart pointer properties
      isSmartPointer,
      pointeeType,
      sharingPolicy,
      rawGetPointee,
      rawConstructor,
      rawShare,
      rawDestructor
    ) {
      this.name = name;
      this.registeredClass = registeredClass;
      this.isReference = isReference;
      this.isConst = isConst;
  
      // smart pointer properties
      this.isSmartPointer = isSmartPointer;
      this.pointeeType = pointeeType;
      this.sharingPolicy = sharingPolicy;
      this.rawGetPointee = rawGetPointee;
      this.rawConstructor = rawConstructor;
      this.rawShare = rawShare;
      this.rawDestructor = rawDestructor;
  
      if (!isSmartPointer && registeredClass.baseClass === undefined) {
        if (isConst) {
          this['toWireType'] = constNoSmartPtrRawPointerToWireType;
          this.destructorFunction = null;
        } else {
          this['toWireType'] = nonConstNoSmartPtrRawPointerToWireType;
          this.destructorFunction = null;
        }
      } else {
        this['toWireType'] = genericPointerToWireType;
        // Here we must leave this.destructorFunction undefined, since whether genericPointerToWireType returns
        // a pointer that needs to be freed up is runtime-dependent, and cannot be evaluated at registration time.
        // TODO: Create an alternative mechanism that allows removing the use of var destructors = []; array in
        //       craftInvokerFunction altogether.
      }
    }
  
  /** @param {number=} numArguments */
  function replacePublicSymbol(name, value, numArguments) {
      if (!Module.hasOwnProperty(name)) {
        throwInternalError('Replacing nonexistant public symbol');
      }
      // If there's an overload table for this symbol, replace the symbol in the overload table instead.
      if (undefined !== Module[name].overloadTable && undefined !== numArguments) {
        Module[name].overloadTable[numArguments] = value;
      }
      else {
        Module[name] = value;
        Module[name].argCount = numArguments;
      }
    }
  
  
  
  function dynCallLegacy(sig, ptr, args) {
      assert(('dynCall_' + sig) in Module, 'bad function pointer type - dynCall function not found for sig \'' + sig + '\'');
      if (args && args.length) {
        // j (64-bit integer) must be passed in as two numbers [low 32, high 32].
        assert(args.length === sig.substring(1).replace(/j/g, '--').length);
      } else {
        assert(sig.length == 1);
      }
      var f = Module['dynCall_' + sig];
      return args && args.length ? f.apply(null, [ptr].concat(args)) : f.call(null, ptr);
    }
  
  var wasmTableMirror = [];
  
  function getWasmTableEntry(funcPtr) {
      var func = wasmTableMirror[funcPtr];
      if (!func) {
        if (funcPtr >= wasmTableMirror.length) wasmTableMirror.length = funcPtr + 1;
        wasmTableMirror[funcPtr] = func = wasmTable.get(funcPtr);
      }
      assert(wasmTable.get(funcPtr) == func, "JavaScript-side Wasm function table mirror is out of date!");
      return func;
    }
  
  /** @param {Object=} args */
  function dynCall(sig, ptr, args) {
      // Without WASM_BIGINT support we cannot directly call function with i64 as
      // part of thier signature, so we rely the dynCall functions generated by
      // wasm-emscripten-finalize
      if (sig.includes('j')) {
        return dynCallLegacy(sig, ptr, args);
      }
      assert(getWasmTableEntry(ptr), 'missing table entry in dynCall: ' + ptr);
      var rtn = getWasmTableEntry(ptr).apply(null, args);
      return rtn;
    }
  
  function getDynCaller(sig, ptr) {
      assert(sig.includes('j') || sig.includes('p'), 'getDynCaller should only be called with i64 sigs');
      var argCache = [];
      return function() {
        argCache.length = 0;
        Object.assign(argCache, arguments);
        return dynCall(sig, ptr, argCache);
      };
    }
  
  
  function embind__requireFunction(signature, rawFunction) {
      signature = readLatin1String(signature);
  
      function makeDynCaller() {
        if (signature.includes('j')) {
          return getDynCaller(signature, rawFunction);
        }
        return getWasmTableEntry(rawFunction);
      }
  
      var fp = makeDynCaller();
      if (typeof fp != "function") {
          throwBindingError("unknown function pointer with signature " + signature + ": " + rawFunction);
      }
      return fp;
    }
  
  
  
  var UnboundTypeError = undefined;
  
  
  function getTypeName(type) {
      var ptr = ___getTypeName(type);
      var rv = readLatin1String(ptr);
      _free(ptr);
      return rv;
    }
  function throwUnboundTypeError(message, types) {
      var unboundTypes = [];
      var seen = {};
      function visit(type) {
        if (seen[type]) {
          return;
        }
        if (registeredTypes[type]) {
          return;
        }
        if (typeDependencies[type]) {
          typeDependencies[type].forEach(visit);
          return;
        }
        unboundTypes.push(type);
        seen[type] = true;
      }
      types.forEach(visit);
  
      throw new UnboundTypeError(message + ': ' + unboundTypes.map(getTypeName).join([', ']));
    }
  
  function __embind_register_class(rawType,
                                     rawPointerType,
                                     rawConstPointerType,
                                     baseClassRawType,
                                     getActualTypeSignature,
                                     getActualType,
                                     upcastSignature,
                                     upcast,
                                     downcastSignature,
                                     downcast,
                                     name,
                                     destructorSignature,
                                     rawDestructor) {
      name = readLatin1String(name);
      getActualType = embind__requireFunction(getActualTypeSignature, getActualType);
      if (upcast) {
        upcast = embind__requireFunction(upcastSignature, upcast);
      }
      if (downcast) {
        downcast = embind__requireFunction(downcastSignature, downcast);
      }
      rawDestructor = embind__requireFunction(destructorSignature, rawDestructor);
      var legalFunctionName = makeLegalFunctionName(name);
  
      exposePublicSymbol(legalFunctionName, function() {
        // this code cannot run if baseClassRawType is zero
        throwUnboundTypeError('Cannot construct ' + name + ' due to unbound types', [baseClassRawType]);
      });
  
      whenDependentTypesAreResolved(
        [rawType, rawPointerType, rawConstPointerType],
        baseClassRawType ? [baseClassRawType] : [],
        function(base) {
          base = base[0];
  
          var baseClass;
          var basePrototype;
          if (baseClassRawType) {
            baseClass = base.registeredClass;
            basePrototype = baseClass.instancePrototype;
          } else {
            basePrototype = ClassHandle.prototype;
          }
  
          var constructor = createNamedFunction(legalFunctionName, function() {
            if (Object.getPrototypeOf(this) !== instancePrototype) {
              throw new BindingError("Use 'new' to construct " + name);
            }
            if (undefined === registeredClass.constructor_body) {
              throw new BindingError(name + " has no accessible constructor");
            }
            var body = registeredClass.constructor_body[arguments.length];
            if (undefined === body) {
              throw new BindingError("Tried to invoke ctor of " + name + " with invalid number of parameters (" + arguments.length + ") - expected (" + Object.keys(registeredClass.constructor_body).toString() + ") parameters instead!");
            }
            return body.apply(this, arguments);
          });
  
          var instancePrototype = Object.create(basePrototype, {
            constructor: { value: constructor },
          });
  
          constructor.prototype = instancePrototype;
  
          var registeredClass = new RegisteredClass(name,
                                                    constructor,
                                                    instancePrototype,
                                                    rawDestructor,
                                                    baseClass,
                                                    getActualType,
                                                    upcast,
                                                    downcast);
  
          var referenceConverter = new RegisteredPointer(name,
                                                         registeredClass,
                                                         true,
                                                         false,
                                                         false);
  
          var pointerConverter = new RegisteredPointer(name + '*',
                                                       registeredClass,
                                                       false,
                                                       false,
                                                       false);
  
          var constPointerConverter = new RegisteredPointer(name + ' const*',
                                                            registeredClass,
                                                            false,
                                                            true,
                                                            false);
  
          registeredPointers[rawType] = {
            pointerType: pointerConverter,
            constPointerType: constPointerConverter
          };
  
          replacePublicSymbol(legalFunctionName, constructor);
  
          return [referenceConverter, pointerConverter, constPointerConverter];
        }
      );
    }

  
  function runDestructors(destructors) {
      while (destructors.length) {
        var ptr = destructors.pop();
        var del = destructors.pop();
        del(ptr);
      }
    }
  
  
  function newFunc(constructor, argumentList) {
      if (!(constructor instanceof Function)) {
        throw new TypeError('new_ called with constructor type ' + typeof(constructor) + " which is not a function");
      }
      /*
       * Previously, the following line was just:
       *   function dummy() {};
       * Unfortunately, Chrome was preserving 'dummy' as the object's name, even
       * though at creation, the 'dummy' has the correct constructor name.  Thus,
       * objects created with IMVU.new would show up in the debugger as 'dummy',
       * which isn't very helpful.  Using IMVU.createNamedFunction addresses the
       * issue.  Doublely-unfortunately, there's no way to write a test for this
       * behavior.  -NRD 2013.02.22
       */
      var dummy = createNamedFunction(constructor.name || 'unknownFunctionName', function(){});
      dummy.prototype = constructor.prototype;
      var obj = new dummy;
  
      var r = constructor.apply(obj, argumentList);
      return (r instanceof Object) ? r : obj;
    }
  function craftInvokerFunction(humanName, argTypes, classType, cppInvokerFunc, cppTargetFunc, isAsync) {
      // humanName: a human-readable string name for the function to be generated.
      // argTypes: An array that contains the embind type objects for all types in the function signature.
      //    argTypes[0] is the type object for the function return value.
      //    argTypes[1] is the type object for function this object/class type, or null if not crafting an invoker for a class method.
      //    argTypes[2...] are the actual function parameters.
      // classType: The embind type object for the class to be bound, or null if this is not a method of a class.
      // cppInvokerFunc: JS Function object to the C++-side function that interops into C++ code.
      // cppTargetFunc: Function pointer (an integer to FUNCTION_TABLE) to the target C++ function the cppInvokerFunc will end up calling.
      var argCount = argTypes.length;
  
      if (argCount < 2) {
        throwBindingError("argTypes array size mismatch! Must at least get return value and 'this' types!");
      }
  
      assert(!isAsync, 'Async bindings are only supported with JSPI.');
  
      var isClassMethodFunc = (argTypes[1] !== null && classType !== null);
  
      // Free functions with signature "void function()" do not need an invoker that marshalls between wire types.
  // TODO: This omits argument count check - enable only at -O3 or similar.
  //    if (ENABLE_UNSAFE_OPTS && argCount == 2 && argTypes[0].name == "void" && !isClassMethodFunc) {
  //       return FUNCTION_TABLE[fn];
  //    }
  
      // Determine if we need to use a dynamic stack to store the destructors for the function parameters.
      // TODO: Remove this completely once all function invokers are being dynamically generated.
      var needsDestructorStack = false;
  
      for (var i = 1; i < argTypes.length; ++i) { // Skip return value at index 0 - it's not deleted here.
        if (argTypes[i] !== null && argTypes[i].destructorFunction === undefined) { // The type does not define a destructor function - must use dynamic stack
          needsDestructorStack = true;
          break;
        }
      }
  
      var returns = (argTypes[0].name !== "void");
  
      var argsList = "";
      var argsListWired = "";
      for (var i = 0; i < argCount - 2; ++i) {
        argsList += (i!==0?", ":"")+"arg"+i;
        argsListWired += (i!==0?", ":"")+"arg"+i+"Wired";
      }
  
      var invokerFnBody =
          "return function "+makeLegalFunctionName(humanName)+"("+argsList+") {\n" +
          "if (arguments.length !== "+(argCount - 2)+") {\n" +
              "throwBindingError('function "+humanName+" called with ' + arguments.length + ' arguments, expected "+(argCount - 2)+" args!');\n" +
          "}\n";
  
      if (needsDestructorStack) {
        invokerFnBody += "var destructors = [];\n";
      }
  
      var dtorStack = needsDestructorStack ? "destructors" : "null";
      var args1 = ["throwBindingError", "invoker", "fn", "runDestructors", "retType", "classParam"];
      var args2 = [throwBindingError, cppInvokerFunc, cppTargetFunc, runDestructors, argTypes[0], argTypes[1]];
  
      if (isClassMethodFunc) {
        invokerFnBody += "var thisWired = classParam.toWireType("+dtorStack+", this);\n";
      }
  
      for (var i = 0; i < argCount - 2; ++i) {
        invokerFnBody += "var arg"+i+"Wired = argType"+i+".toWireType("+dtorStack+", arg"+i+"); // "+argTypes[i+2].name+"\n";
        args1.push("argType"+i);
        args2.push(argTypes[i+2]);
      }
  
      if (isClassMethodFunc) {
        argsListWired = "thisWired" + (argsListWired.length > 0 ? ", " : "") + argsListWired;
      }
  
      invokerFnBody +=
          (returns || isAsync ? "var rv = ":"") + "invoker(fn"+(argsListWired.length>0?", ":"")+argsListWired+");\n";
  
      if (needsDestructorStack) {
        invokerFnBody += "runDestructors(destructors);\n";
      } else {
        for (var i = isClassMethodFunc?1:2; i < argTypes.length; ++i) { // Skip return value at index 0 - it's not deleted here. Also skip class type if not a method.
          var paramName = (i === 1 ? "thisWired" : ("arg"+(i - 2)+"Wired"));
          if (argTypes[i].destructorFunction !== null) {
            invokerFnBody += paramName+"_dtor("+paramName+"); // "+argTypes[i].name+"\n";
            args1.push(paramName+"_dtor");
            args2.push(argTypes[i].destructorFunction);
          }
        }
      }
  
      if (returns) {
        invokerFnBody += "var ret = retType.fromWireType(rv);\n" +
                         "return ret;\n";
      }
  
      invokerFnBody += "}\n";
  
      args1.push(invokerFnBody);
  
      return newFunc(Function, args1).apply(null, args2);
    }
  
  
  function heap32VectorToArray(count, firstElement) {
      var array = [];
      for (var i = 0; i < count; i++) {
          // TODO(https://github.com/emscripten-core/emscripten/issues/17310):
          // Find a way to hoist the `>> 2` or `>> 3` out of this loop.
          array.push(HEAPU32[(((firstElement)+(i * 4))>>2)]);
      }
      return array;
    }
  
  
  
  
  function __embind_register_class_class_function(rawClassType,
                                                    methodName,
                                                    argCount,
                                                    rawArgTypesAddr,
                                                    invokerSignature,
                                                    rawInvoker,
                                                    fn,
                                                    isAsync) {
      var rawArgTypes = heap32VectorToArray(argCount, rawArgTypesAddr);
      methodName = readLatin1String(methodName);
      rawInvoker = embind__requireFunction(invokerSignature, rawInvoker);
      whenDependentTypesAreResolved([], [rawClassType], function(classType) {
        classType = classType[0];
        var humanName = classType.name + '.' + methodName;
  
        function unboundTypesHandler() {
          throwUnboundTypeError('Cannot call ' + humanName + ' due to unbound types', rawArgTypes);
        }
  
        if (methodName.startsWith("@@")) {
          methodName = Symbol[methodName.substring(2)];
        }
  
        var proto = classType.registeredClass.constructor;
        if (undefined === proto[methodName]) {
          // This is the first function to be registered with this name.
          unboundTypesHandler.argCount = argCount-1;
          proto[methodName] = unboundTypesHandler;
        } else {
          // There was an existing function with the same name registered. Set up
          // a function overload routing table.
          ensureOverloadTable(proto, methodName, humanName);
          proto[methodName].overloadTable[argCount-1] = unboundTypesHandler;
        }
  
        whenDependentTypesAreResolved([], rawArgTypes, function(argTypes) {
          // Replace the initial unbound-types-handler stub with the proper
          // function. If multiple overloads are registered, the function handlers
          // go into an overload table.
          var invokerArgsArray = [argTypes[0] /* return value */, null /* no class 'this'*/].concat(argTypes.slice(1) /* actual params */);
          var func = craftInvokerFunction(humanName, invokerArgsArray, null /* no class 'this'*/, rawInvoker, fn, isAsync);
          if (undefined === proto[methodName].overloadTable) {
            func.argCount = argCount-1;
            proto[methodName] = func;
          } else {
            proto[methodName].overloadTable[argCount-1] = func;
          }
          return [];
        });
        return [];
      });
    }

  
  
  
  
  
  
  function __embind_register_class_constructor(
      rawClassType,
      argCount,
      rawArgTypesAddr,
      invokerSignature,
      invoker,
      rawConstructor
    ) {
      assert(argCount > 0);
      var rawArgTypes = heap32VectorToArray(argCount, rawArgTypesAddr);
      invoker = embind__requireFunction(invokerSignature, invoker);
  
      whenDependentTypesAreResolved([], [rawClassType], function(classType) {
        classType = classType[0];
        var humanName = 'constructor ' + classType.name;
  
        if (undefined === classType.registeredClass.constructor_body) {
          classType.registeredClass.constructor_body = [];
        }
        if (undefined !== classType.registeredClass.constructor_body[argCount - 1]) {
          throw new BindingError("Cannot register multiple constructors with identical number of parameters (" + (argCount-1) + ") for class '" + classType.name + "'! Overload resolution is currently only performed using the parameter count, not actual type info!");
        }
        classType.registeredClass.constructor_body[argCount - 1] = () => {
          throwUnboundTypeError('Cannot construct ' + classType.name + ' due to unbound types', rawArgTypes);
        };
  
        whenDependentTypesAreResolved([], rawArgTypes, function(argTypes) {
          // Insert empty slot for context type (argTypes[1]).
          argTypes.splice(1, 0, null);
          classType.registeredClass.constructor_body[argCount - 1] = craftInvokerFunction(humanName, argTypes, null, invoker, rawConstructor);
          return [];
        });
        return [];
      });
    }

  
  
  
  
  
  function __embind_register_class_function(rawClassType,
                                              methodName,
                                              argCount,
                                              rawArgTypesAddr, // [ReturnType, ThisType, Args...]
                                              invokerSignature,
                                              rawInvoker,
                                              context,
                                              isPureVirtual,
                                              isAsync) {
      var rawArgTypes = heap32VectorToArray(argCount, rawArgTypesAddr);
      methodName = readLatin1String(methodName);
      rawInvoker = embind__requireFunction(invokerSignature, rawInvoker);
  
      whenDependentTypesAreResolved([], [rawClassType], function(classType) {
        classType = classType[0];
        var humanName = classType.name + '.' + methodName;
  
        if (methodName.startsWith("@@")) {
          methodName = Symbol[methodName.substring(2)];
        }
  
        if (isPureVirtual) {
          classType.registeredClass.pureVirtualFunctions.push(methodName);
        }
  
        function unboundTypesHandler() {
          throwUnboundTypeError('Cannot call ' + humanName + ' due to unbound types', rawArgTypes);
        }
  
        var proto = classType.registeredClass.instancePrototype;
        var method = proto[methodName];
        if (undefined === method || (undefined === method.overloadTable && method.className !== classType.name && method.argCount === argCount - 2)) {
          // This is the first overload to be registered, OR we are replacing a
          // function in the base class with a function in the derived class.
          unboundTypesHandler.argCount = argCount - 2;
          unboundTypesHandler.className = classType.name;
          proto[methodName] = unboundTypesHandler;
        } else {
          // There was an existing function with the same name registered. Set up
          // a function overload routing table.
          ensureOverloadTable(proto, methodName, humanName);
          proto[methodName].overloadTable[argCount - 2] = unboundTypesHandler;
        }
  
        whenDependentTypesAreResolved([], rawArgTypes, function(argTypes) {
          var memberFunction = craftInvokerFunction(humanName, argTypes, classType, rawInvoker, context, isAsync);
  
          // Replace the initial unbound-handler-stub function with the appropriate member function, now that all types
          // are resolved. If multiple overloads are registered for this function, the function goes into an overload table.
          if (undefined === proto[methodName].overloadTable) {
            // Set argCount in case an overload is registered later
            memberFunction.argCount = argCount - 2;
            proto[methodName] = memberFunction;
          } else {
            proto[methodName].overloadTable[argCount - 2] = memberFunction;
          }
  
          return [];
        });
        return [];
      });
    }

  /** @constructor */
  function HandleAllocator() {
      // Reserve slot 0 so that 0 is always an invalid handle
      this.allocated = [undefined];
      this.freelist = [];
      this.get = function(id) {
        assert(this.allocated[id] !== undefined, 'invalid handle: ' + id);
        return this.allocated[id];
      };
      this.allocate = function(handle) {
        let id = this.freelist.pop() || this.allocated.length;
        this.allocated[id] = handle;
        return id;
      };
      this.free = function(id) {
        assert(this.allocated[id] !== undefined);
        // Set the slot to `undefined` rather than using `delete` here since
        // apparently arrays with holes in them can be less efficient.
        this.allocated[id] = undefined;
        this.freelist.push(id);
      };
    }
  var emval_handles = new HandleAllocator();  function __emval_decref(handle) {
      if (handle >= emval_handles.reserved && 0 === --emval_handles.get(handle).refcount) {
        emval_handles.free(handle);
      }
    }
  
  
  
  function count_emval_handles() {
      var count = 0;
      for (var i = emval_handles.reserved; i < emval_handles.allocated.length; ++i) {
        if (emval_handles.allocated[i] !== undefined) {
          ++count;
        }
      }
      return count;
    }
  
  function init_emval() {
      // reserve some special values. These never get de-allocated.
      // The HandleAllocator takes care of reserving zero.
      emval_handles.allocated.push(
        {value: undefined},
        {value: null},
        {value: true},
        {value: false},
      );
      emval_handles.reserved = emval_handles.allocated.length;
      Module['count_emval_handles'] = count_emval_handles;
    }
  var Emval = {toValue:(handle) => {
        if (!handle) {
            throwBindingError('Cannot use deleted val. handle = ' + handle);
        }
        return emval_handles.get(handle).value;
      },toHandle:(value) => {
        switch (value) {
          case undefined: return 1;
          case null: return 2;
          case true: return 3;
          case false: return 4;
          default:{
            return emval_handles.allocate({refcount: 1, value: value});
          }
        }
      }};
  
  
  
  function __embind_register_emval(rawType, name) {
      name = readLatin1String(name);
      registerType(rawType, {
        name: name,
        'fromWireType': function(handle) {
          var rv = Emval.toValue(handle);
          __emval_decref(handle);
          return rv;
        },
        'toWireType': function(destructors, value) {
          return Emval.toHandle(value);
        },
        'argPackAdvance': 8,
        'readValueFromPointer': simpleReadValueFromPointer,
        destructorFunction: null, // This type does not need a destructor
  
        // TODO: do we need a deleteObject here?  write a test where
        // emval is passed into JS via an interface
      });
    }

  function embindRepr(v) {
      if (v === null) {
          return 'null';
      }
      var t = typeof v;
      if (t === 'object' || t === 'array' || t === 'function') {
          return v.toString();
      } else {
          return '' + v;
      }
    }
  
  function floatReadValueFromPointer(name, shift) {
      switch (shift) {
          case 2: return function(pointer) {
              return this['fromWireType'](HEAPF32[pointer >> 2]);
          };
          case 3: return function(pointer) {
              return this['fromWireType'](HEAPF64[pointer >> 3]);
          };
          default:
              throw new TypeError("Unknown float type: " + name);
      }
    }
  
  
  
  function __embind_register_float(rawType, name, size) {
      var shift = getShiftFromSize(size);
      name = readLatin1String(name);
      registerType(rawType, {
        name: name,
        'fromWireType': function(value) {
           return value;
        },
        'toWireType': function(destructors, value) {
          if (typeof value != "number" && typeof value != "boolean") {
            throw new TypeError('Cannot convert "' + embindRepr(value) + '" to ' + this.name);
          }
          // The VM will perform JS to Wasm value conversion, according to the spec:
          // https://www.w3.org/TR/wasm-js-api-1/#towebassemblyvalue
          return value;
        },
        'argPackAdvance': 8,
        'readValueFromPointer': floatReadValueFromPointer(name, shift),
        destructorFunction: null, // This type does not need a destructor
      });
    }

  
  
  
  
  
  
  
  function __embind_register_function(name, argCount, rawArgTypesAddr, signature, rawInvoker, fn, isAsync) {
      var argTypes = heap32VectorToArray(argCount, rawArgTypesAddr);
      name = readLatin1String(name);
  
      rawInvoker = embind__requireFunction(signature, rawInvoker);
  
      exposePublicSymbol(name, function() {
        throwUnboundTypeError('Cannot call ' + name + ' due to unbound types', argTypes);
      }, argCount - 1);
  
      whenDependentTypesAreResolved([], argTypes, function(argTypes) {
        var invokerArgsArray = [argTypes[0] /* return value */, null /* no class 'this'*/].concat(argTypes.slice(1) /* actual params */);
        replacePublicSymbol(name, craftInvokerFunction(name, invokerArgsArray, null /* no class 'this'*/, rawInvoker, fn, isAsync), argCount - 1);
        return [];
      });
    }

  
  
  function integerReadValueFromPointer(name, shift, signed) {
      // integers are quite common, so generate very specialized functions
      switch (shift) {
          case 0: return signed ?
              function readS8FromPointer(pointer) { return HEAP8[pointer]; } :
              function readU8FromPointer(pointer) { return HEAPU8[pointer]; };
          case 1: return signed ?
              function readS16FromPointer(pointer) { return HEAP16[pointer >> 1]; } :
              function readU16FromPointer(pointer) { return HEAPU16[pointer >> 1]; };
          case 2: return signed ?
              function readS32FromPointer(pointer) { return HEAP32[pointer >> 2]; } :
              function readU32FromPointer(pointer) { return HEAPU32[pointer >> 2]; };
          default:
              throw new TypeError("Unknown integer type: " + name);
      }
    }
  
  
  function __embind_register_integer(primitiveType, name, size, minRange, maxRange) {
      name = readLatin1String(name);
      // LLVM doesn't have signed and unsigned 32-bit types, so u32 literals come
      // out as 'i32 -1'. Always treat those as max u32.
      if (maxRange === -1) {
          maxRange = 4294967295;
      }
  
      var shift = getShiftFromSize(size);
  
      var fromWireType = (value) => value;
  
      if (minRange === 0) {
          var bitshift = 32 - 8*size;
          fromWireType = (value) => (value << bitshift) >>> bitshift;
      }
  
      var isUnsignedType = (name.includes('unsigned'));
      var checkAssertions = (value, toTypeName) => {
        if (typeof value != "number" && typeof value != "boolean") {
          throw new TypeError('Cannot convert "' + embindRepr(value) + '" to ' + toTypeName);
        }
        if (value < minRange || value > maxRange) {
          throw new TypeError('Passing a number "' + embindRepr(value) + '" from JS side to C/C++ side to an argument of type "' + name + '", which is outside the valid range [' + minRange + ', ' + maxRange + ']!');
        }
      };
      var toWireType;
      if (isUnsignedType) {
        toWireType = function(destructors, value) {
          checkAssertions(value, this.name);
          return value >>> 0;
        };
      } else {
        toWireType = function(destructors, value) {
          checkAssertions(value, this.name);
          // The VM will perform JS to Wasm value conversion, according to the spec:
          // https://www.w3.org/TR/wasm-js-api-1/#towebassemblyvalue
          return value;
        };
      }
      registerType(primitiveType, {
        name: name,
        'fromWireType': fromWireType,
        'toWireType': toWireType,
        'argPackAdvance': 8,
        'readValueFromPointer': integerReadValueFromPointer(name, shift, minRange !== 0),
        destructorFunction: null, // This type does not need a destructor
      });
    }

  
  function __embind_register_memory_view(rawType, dataTypeIndex, name) {
      var typeMapping = [
        Int8Array,
        Uint8Array,
        Int16Array,
        Uint16Array,
        Int32Array,
        Uint32Array,
        Float32Array,
        Float64Array,
      ];
  
      var TA = typeMapping[dataTypeIndex];
  
      function decodeMemoryView(handle) {
        handle = handle >> 2;
        var heap = HEAPU32;
        var size = heap[handle]; // in elements
        var data = heap[handle + 1]; // byte offset into emscripten heap
        return new TA(heap.buffer, data, size);
      }
  
      name = readLatin1String(name);
      registerType(rawType, {
        name: name,
        'fromWireType': decodeMemoryView,
        'argPackAdvance': 8,
        'readValueFromPointer': decodeMemoryView,
      }, {
        ignoreDuplicateRegistrations: true,
      });
    }

  
  
  
  
  
  
  function __embind_register_std_string(rawType, name) {
      name = readLatin1String(name);
      var stdStringIsUTF8
      //process only std::string bindings with UTF8 support, in contrast to e.g. std::basic_string<unsigned char>
      = (name === "std::string");
  
      registerType(rawType, {
        name: name,
        'fromWireType': function(value) {
          var length = HEAPU32[((value)>>2)];
          var payload = value + 4;
  
          var str;
          if (stdStringIsUTF8) {
            var decodeStartPtr = payload;
            // Looping here to support possible embedded '0' bytes
            for (var i = 0; i <= length; ++i) {
              var currentBytePtr = payload + i;
              if (i == length || HEAPU8[currentBytePtr] == 0) {
                var maxRead = currentBytePtr - decodeStartPtr;
                var stringSegment = UTF8ToString(decodeStartPtr, maxRead);
                if (str === undefined) {
                  str = stringSegment;
                } else {
                  str += String.fromCharCode(0);
                  str += stringSegment;
                }
                decodeStartPtr = currentBytePtr + 1;
              }
            }
          } else {
            var a = new Array(length);
            for (var i = 0; i < length; ++i) {
              a[i] = String.fromCharCode(HEAPU8[payload + i]);
            }
            str = a.join('');
          }
  
          _free(value);
  
          return str;
        },
        'toWireType': function(destructors, value) {
          if (value instanceof ArrayBuffer) {
            value = new Uint8Array(value);
          }
  
          var length;
          var valueIsOfTypeString = (typeof value == 'string');
  
          if (!(valueIsOfTypeString || value instanceof Uint8Array || value instanceof Uint8ClampedArray || value instanceof Int8Array)) {
            throwBindingError('Cannot pass non-string to std::string');
          }
          if (stdStringIsUTF8 && valueIsOfTypeString) {
            length = lengthBytesUTF8(value);
          } else {
            length = value.length;
          }
  
          // assumes 4-byte alignment
          var base = _malloc(4 + length + 1);
          var ptr = base + 4;
          HEAPU32[((base)>>2)] = length;
          if (stdStringIsUTF8 && valueIsOfTypeString) {
            stringToUTF8(value, ptr, length + 1);
          } else {
            if (valueIsOfTypeString) {
              for (var i = 0; i < length; ++i) {
                var charCode = value.charCodeAt(i);
                if (charCode > 255) {
                  _free(ptr);
                  throwBindingError('String has UTF-16 code units that do not fit in 8 bits');
                }
                HEAPU8[ptr + i] = charCode;
              }
            } else {
              for (var i = 0; i < length; ++i) {
                HEAPU8[ptr + i] = value[i];
              }
            }
          }
  
          if (destructors !== null) {
            destructors.push(_free, base);
          }
          return base;
        },
        'argPackAdvance': 8,
        'readValueFromPointer': simpleReadValueFromPointer,
        destructorFunction: function(ptr) { _free(ptr); },
      });
    }

  
  
  
  var UTF16Decoder = typeof TextDecoder != 'undefined' ? new TextDecoder('utf-16le') : undefined;  function UTF16ToString(ptr, maxBytesToRead) {
      assert(ptr % 2 == 0, 'Pointer passed to UTF16ToString must be aligned to two bytes!');
      var endPtr = ptr;
      // TextDecoder needs to know the byte length in advance, it doesn't stop on
      // null terminator by itself.
      // Also, use the length info to avoid running tiny strings through
      // TextDecoder, since .subarray() allocates garbage.
      var idx = endPtr >> 1;
      var maxIdx = idx + maxBytesToRead / 2;
      // If maxBytesToRead is not passed explicitly, it will be undefined, and this
      // will always evaluate to true. This saves on code size.
      while (!(idx >= maxIdx) && HEAPU16[idx]) ++idx;
      endPtr = idx << 1;
  
      if (endPtr - ptr > 32 && UTF16Decoder)
        return UTF16Decoder.decode(HEAPU8.subarray(ptr, endPtr));
  
      // Fallback: decode without UTF16Decoder
      var str = '';
  
      // If maxBytesToRead is not passed explicitly, it will be undefined, and the
      // for-loop's condition will always evaluate to true. The loop is then
      // terminated on the first null char.
      for (var i = 0; !(i >= maxBytesToRead / 2); ++i) {
        var codeUnit = HEAP16[(((ptr)+(i*2))>>1)];
        if (codeUnit == 0) break;
        // fromCharCode constructs a character from a UTF-16 code unit, so we can
        // pass the UTF16 string right through.
        str += String.fromCharCode(codeUnit);
      }
  
      return str;
    }
  
  function stringToUTF16(str, outPtr, maxBytesToWrite) {
      assert(outPtr % 2 == 0, 'Pointer passed to stringToUTF16 must be aligned to two bytes!');
      assert(typeof maxBytesToWrite == 'number', 'stringToUTF16(str, outPtr, maxBytesToWrite) is missing the third parameter that specifies the length of the output buffer!');
      // Backwards compatibility: if max bytes is not specified, assume unsafe unbounded write is allowed.
      if (maxBytesToWrite === undefined) {
        maxBytesToWrite = 0x7FFFFFFF;
      }
      if (maxBytesToWrite < 2) return 0;
      maxBytesToWrite -= 2; // Null terminator.
      var startPtr = outPtr;
      var numCharsToWrite = (maxBytesToWrite < str.length*2) ? (maxBytesToWrite / 2) : str.length;
      for (var i = 0; i < numCharsToWrite; ++i) {
        // charCodeAt returns a UTF-16 encoded code unit, so it can be directly written to the HEAP.
        var codeUnit = str.charCodeAt(i); // possibly a lead surrogate
        HEAP16[((outPtr)>>1)] = codeUnit;
        outPtr += 2;
      }
      // Null-terminate the pointer to the HEAP.
      HEAP16[((outPtr)>>1)] = 0;
      return outPtr - startPtr;
    }
  
  function lengthBytesUTF16(str) {
      return str.length*2;
    }
  
  function UTF32ToString(ptr, maxBytesToRead) {
      assert(ptr % 4 == 0, 'Pointer passed to UTF32ToString must be aligned to four bytes!');
      var i = 0;
  
      var str = '';
      // If maxBytesToRead is not passed explicitly, it will be undefined, and this
      // will always evaluate to true. This saves on code size.
      while (!(i >= maxBytesToRead / 4)) {
        var utf32 = HEAP32[(((ptr)+(i*4))>>2)];
        if (utf32 == 0) break;
        ++i;
        // Gotcha: fromCharCode constructs a character from a UTF-16 encoded code (pair), not from a Unicode code point! So encode the code point to UTF-16 for constructing.
        // See http://unicode.org/faq/utf_bom.html#utf16-3
        if (utf32 >= 0x10000) {
          var ch = utf32 - 0x10000;
          str += String.fromCharCode(0xD800 | (ch >> 10), 0xDC00 | (ch & 0x3FF));
        } else {
          str += String.fromCharCode(utf32);
        }
      }
      return str;
    }
  
  function stringToUTF32(str, outPtr, maxBytesToWrite) {
      assert(outPtr % 4 == 0, 'Pointer passed to stringToUTF32 must be aligned to four bytes!');
      assert(typeof maxBytesToWrite == 'number', 'stringToUTF32(str, outPtr, maxBytesToWrite) is missing the third parameter that specifies the length of the output buffer!');
      // Backwards compatibility: if max bytes is not specified, assume unsafe unbounded write is allowed.
      if (maxBytesToWrite === undefined) {
        maxBytesToWrite = 0x7FFFFFFF;
      }
      if (maxBytesToWrite < 4) return 0;
      var startPtr = outPtr;
      var endPtr = startPtr + maxBytesToWrite - 4;
      for (var i = 0; i < str.length; ++i) {
        // Gotcha: charCodeAt returns a 16-bit word that is a UTF-16 encoded code unit, not a Unicode code point of the character! We must decode the string to UTF-32 to the heap.
        // See http://unicode.org/faq/utf_bom.html#utf16-3
        var codeUnit = str.charCodeAt(i); // possibly a lead surrogate
        if (codeUnit >= 0xD800 && codeUnit <= 0xDFFF) {
          var trailSurrogate = str.charCodeAt(++i);
          codeUnit = 0x10000 + ((codeUnit & 0x3FF) << 10) | (trailSurrogate & 0x3FF);
        }
        HEAP32[((outPtr)>>2)] = codeUnit;
        outPtr += 4;
        if (outPtr + 4 > endPtr) break;
      }
      // Null-terminate the pointer to the HEAP.
      HEAP32[((outPtr)>>2)] = 0;
      return outPtr - startPtr;
    }
  
  function lengthBytesUTF32(str) {
      var len = 0;
      for (var i = 0; i < str.length; ++i) {
        // Gotcha: charCodeAt returns a 16-bit word that is a UTF-16 encoded code unit, not a Unicode code point of the character! We must decode the string to UTF-32 to the heap.
        // See http://unicode.org/faq/utf_bom.html#utf16-3
        var codeUnit = str.charCodeAt(i);
        if (codeUnit >= 0xD800 && codeUnit <= 0xDFFF) ++i; // possibly a lead surrogate, so skip over the tail surrogate.
        len += 4;
      }
  
      return len;
    }
  function __embind_register_std_wstring(rawType, charSize, name) {
      name = readLatin1String(name);
      var decodeString, encodeString, getHeap, lengthBytesUTF, shift;
      if (charSize === 2) {
        decodeString = UTF16ToString;
        encodeString = stringToUTF16;
        lengthBytesUTF = lengthBytesUTF16;
        getHeap = () => HEAPU16;
        shift = 1;
      } else if (charSize === 4) {
        decodeString = UTF32ToString;
        encodeString = stringToUTF32;
        lengthBytesUTF = lengthBytesUTF32;
        getHeap = () => HEAPU32;
        shift = 2;
      }
      registerType(rawType, {
        name: name,
        'fromWireType': function(value) {
          // Code mostly taken from _embind_register_std_string fromWireType
          var length = HEAPU32[value >> 2];
          var HEAP = getHeap();
          var str;
  
          var decodeStartPtr = value + 4;
          // Looping here to support possible embedded '0' bytes
          for (var i = 0; i <= length; ++i) {
            var currentBytePtr = value + 4 + i * charSize;
            if (i == length || HEAP[currentBytePtr >> shift] == 0) {
              var maxReadBytes = currentBytePtr - decodeStartPtr;
              var stringSegment = decodeString(decodeStartPtr, maxReadBytes);
              if (str === undefined) {
                str = stringSegment;
              } else {
                str += String.fromCharCode(0);
                str += stringSegment;
              }
              decodeStartPtr = currentBytePtr + charSize;
            }
          }
  
          _free(value);
  
          return str;
        },
        'toWireType': function(destructors, value) {
          if (!(typeof value == 'string')) {
            throwBindingError('Cannot pass non-string to C++ string type ' + name);
          }
  
          // assumes 4-byte alignment
          var length = lengthBytesUTF(value);
          var ptr = _malloc(4 + length + charSize);
          HEAPU32[ptr >> 2] = length >> shift;
  
          encodeString(value, ptr + 4, length + charSize);
  
          if (destructors !== null) {
            destructors.push(_free, ptr);
          }
          return ptr;
        },
        'argPackAdvance': 8,
        'readValueFromPointer': simpleReadValueFromPointer,
        destructorFunction: function(ptr) { _free(ptr); },
      });
    }

  
  function __embind_register_void(rawType, name) {
      name = readLatin1String(name);
      registerType(rawType, {
          isVoid: true, // void return values can be optimized out sometimes
          name: name,
          'argPackAdvance': 0,
          'fromWireType': function() {
              return undefined;
          },
          'toWireType': function(destructors, o) {
              // TODO: assert if anything else is given?
              return undefined;
          },
      });
    }

  
  
  
  function requireRegisteredType(rawType, humanName) {
      var impl = registeredTypes[rawType];
      if (undefined === impl) {
          throwBindingError(humanName + " has unknown type " + getTypeName(rawType));
      }
      return impl;
    }
  function __emval_as(handle, returnType, destructorsRef) {
      handle = Emval.toValue(handle);
      returnType = requireRegisteredType(returnType, 'emval::as');
      var destructors = [];
      var rd = Emval.toHandle(destructors);
      HEAPU32[((destructorsRef)>>2)] = rd;
      return returnType['toWireType'](destructors, handle);
    }

  function emval_allocateDestructors(destructorsRef) {
      var destructors = [];
      HEAPU32[((destructorsRef)>>2)] = Emval.toHandle(destructors);
      return destructors;
    }
  
  var emval_symbols = {};
  
  function getStringOrSymbol(address) {
      var symbol = emval_symbols[address];
      if (symbol === undefined) {
        return readLatin1String(address);
      }
      return symbol;
    }
  
  var emval_methodCallers = [];
  
  function __emval_call_method(caller, handle, methodName, destructorsRef, args) {
      caller = emval_methodCallers[caller];
      handle = Emval.toValue(handle);
      methodName = getStringOrSymbol(methodName);
      return caller(handle, methodName, emval_allocateDestructors(destructorsRef), args);
    }

  
  
  
  function __emval_call_void_method(caller, handle, methodName, args) {
      caller = emval_methodCallers[caller];
      handle = Emval.toValue(handle);
      methodName = getStringOrSymbol(methodName);
      caller(handle, methodName, null, args);
    }


  function emval_addMethodCaller(caller) {
      var id = emval_methodCallers.length;
      emval_methodCallers.push(caller);
      return id;
    }
  
  function emval_lookupTypes(argCount, argTypes) {
      var a = new Array(argCount);
      for (var i = 0; i < argCount; ++i) {
        a[i] = requireRegisteredType(HEAPU32[(((argTypes)+(i * 4))>>2)],
                                     "parameter " + i);
      }
      return a;
    }
  
  
  var emval_registeredMethods = [];
  
  function __emval_get_method_caller(argCount, argTypes) {
      var types = emval_lookupTypes(argCount, argTypes);
      var retType = types[0];
      var signatureName = retType.name + "_$" + types.slice(1).map(function (t) { return t.name; }).join("_") + "$";
      var returnId = emval_registeredMethods[signatureName];
      if (returnId !== undefined) {
        return returnId;
      }
  
      var params = ["retType"];
      var args = [retType];
  
      var argsList = ""; // 'arg0, arg1, arg2, ... , argN'
      for (var i = 0; i < argCount - 1; ++i) {
        argsList += (i !== 0 ? ", " : "") + "arg" + i;
        params.push("argType" + i);
        args.push(types[1 + i]);
      }
  
      var functionName = makeLegalFunctionName("methodCaller_" + signatureName);
      var functionBody =
          "return function " + functionName + "(handle, name, destructors, args) {\n";
  
      var offset = 0;
      for (var i = 0; i < argCount - 1; ++i) {
          functionBody +=
          "    var arg" + i + " = argType" + i + ".readValueFromPointer(args" + (offset ? ("+"+offset) : "") + ");\n";
          offset += types[i + 1]['argPackAdvance'];
      }
      functionBody +=
          "    var rv = handle[name](" + argsList + ");\n";
      for (var i = 0; i < argCount - 1; ++i) {
          if (types[i + 1]['deleteObject']) {
              functionBody +=
              "    argType" + i + ".deleteObject(arg" + i + ");\n";
          }
      }
      if (!retType.isVoid) {
          functionBody +=
          "    return retType.toWireType(destructors, rv);\n";
      }
      functionBody +=
          "};\n";
  
      params.push(functionBody);
      var invokerFunction = newFunc(Function, params).apply(null, args);
      returnId = emval_addMethodCaller(invokerFunction);
      emval_registeredMethods[signatureName] = returnId;
      return returnId;
    }

  function __emval_get_property(handle, key) {
      handle = Emval.toValue(handle);
      key = Emval.toValue(key);
      return Emval.toHandle(handle[key]);
    }

  function __emval_incref(handle) {
      if (handle > 4) {
        emval_handles.get(handle).refcount += 1;
      }
    }

  function __emval_new_array() {
      return Emval.toHandle([]);
    }

  function __emval_new_array_from_memory_view(view) {
      view = Emval.toValue(view);
      // using for..loop is faster than Array.from
      var a = new Array(view.length);
      for (var i = 0; i < view.length; i++) a[i] = view[i];
      return Emval.toHandle(a);
    }

  
  function __emval_new_cstring(v) {
      return Emval.toHandle(getStringOrSymbol(v));
    }

  function __emval_new_object() {
      return Emval.toHandle({});
    }

  
  
  function __emval_run_destructors(handle) {
      var destructors = Emval.toValue(handle);
      runDestructors(destructors);
      __emval_decref(handle);
    }

  function __emval_set_property(handle, key, value) {
      handle = Emval.toValue(handle);
      key = Emval.toValue(key);
      value = Emval.toValue(value);
      handle[key] = value;
    }

  
  function __emval_take_value(type, arg) {
      type = requireRegisteredType(type, '_emval_take_value');
      var v = type['readValueFromPointer'](arg);
      return Emval.toHandle(v);
    }

  function _abort() {
      abort('native code called abort()');
    }

  var readEmAsmArgsArray = [];
  function readEmAsmArgs(sigPtr, buf) {
      // Nobody should have mutated _readEmAsmArgsArray underneath us to be something else than an array.
      assert(Array.isArray(readEmAsmArgsArray));
      // The input buffer is allocated on the stack, so it must be stack-aligned.
      assert(buf % 16 == 0);
      readEmAsmArgsArray.length = 0;
      var ch;
      // Most arguments are i32s, so shift the buffer pointer so it is a plain
      // index into HEAP32.
      buf >>= 2;
      while (ch = HEAPU8[sigPtr++]) {
        var chr = String.fromCharCode(ch);
        var validChars = ['d', 'f', 'i'];
        assert(validChars.includes(chr), 'Invalid character ' + ch + '("' + chr + '") in readEmAsmArgs! Use only [' + validChars + '], and do not specify "v" for void return argument.');
        // Floats are always passed as doubles, and doubles and int64s take up 8
        // bytes (two 32-bit slots) in memory, align reads to these:
        buf += (ch != 105/*i*/) & buf;
        readEmAsmArgsArray.push(
          ch == 105/*i*/ ? HEAP32[buf] :
         HEAPF64[buf++ >> 1]
        );
        ++buf;
      }
      return readEmAsmArgsArray;
    }
  function runEmAsmFunction(code, sigPtr, argbuf) {
      var args = readEmAsmArgs(sigPtr, argbuf);
      if (!ASM_CONSTS.hasOwnProperty(code)) abort('No EM_ASM constant found at address ' + code);
      return ASM_CONSTS[code].apply(null, args);
    }
  function _emscripten_asm_const_int(code, sigPtr, argbuf) {
      return runEmAsmFunction(code, sigPtr, argbuf);
    }

  function _emscripten_memcpy_big(dest, src, num) {
      HEAPU8.copyWithin(dest, src, src + num);
    }

  function getHeapMax() {
      // Stay one Wasm page short of 4GB: while e.g. Chrome is able to allocate
      // full 4GB Wasm memories, the size will wrap back to 0 bytes in Wasm side
      // for any code that deals with heap sizes, which would require special
      // casing all heap size related code to treat 0 specially.
      return 2147483648;
    }
  
  function emscripten_realloc_buffer(size) {
      var b = wasmMemory.buffer;
      try {
        // round size grow request up to wasm page size (fixed 64KB per spec)
        wasmMemory.grow((size - b.byteLength + 65535) >>> 16); // .grow() takes a delta compared to the previous size
        updateMemoryViews();
        return 1 /*success*/;
      } catch(e) {
        err('emscripten_realloc_buffer: Attempted to grow heap from ' + b.byteLength  + ' bytes to ' + size + ' bytes, but got error: ' + e);
      }
      // implicit 0 return to save code size (caller will cast "undefined" into 0
      // anyhow)
    }
  function _emscripten_resize_heap(requestedSize) {
      var oldSize = HEAPU8.length;
      requestedSize = requestedSize >>> 0;
      // With multithreaded builds, races can happen (another thread might increase the size
      // in between), so return a failure, and let the caller retry.
      assert(requestedSize > oldSize);
  
      // Memory resize rules:
      // 1.  Always increase heap size to at least the requested size, rounded up
      //     to next page multiple.
      // 2a. If MEMORY_GROWTH_LINEAR_STEP == -1, excessively resize the heap
      //     geometrically: increase the heap size according to
      //     MEMORY_GROWTH_GEOMETRIC_STEP factor (default +20%), At most
      //     overreserve by MEMORY_GROWTH_GEOMETRIC_CAP bytes (default 96MB).
      // 2b. If MEMORY_GROWTH_LINEAR_STEP != -1, excessively resize the heap
      //     linearly: increase the heap size by at least
      //     MEMORY_GROWTH_LINEAR_STEP bytes.
      // 3.  Max size for the heap is capped at 2048MB-WASM_PAGE_SIZE, or by
      //     MAXIMUM_MEMORY, or by ASAN limit, depending on which is smallest
      // 4.  If we were unable to allocate as much memory, it may be due to
      //     over-eager decision to excessively reserve due to (3) above.
      //     Hence if an allocation fails, cut down on the amount of excess
      //     growth, in an attempt to succeed to perform a smaller allocation.
  
      // A limit is set for how much we can grow. We should not exceed that
      // (the wasm binary specifies it, so if we tried, we'd fail anyhow).
      var maxHeapSize = getHeapMax();
      if (requestedSize > maxHeapSize) {
        err('Cannot enlarge memory, asked to go up to ' + requestedSize + ' bytes, but the limit is ' + maxHeapSize + ' bytes!');
        return false;
      }
  
      let alignUp = (x, multiple) => x + (multiple - x % multiple) % multiple;
  
      // Loop through potential heap size increases. If we attempt a too eager
      // reservation that fails, cut down on the attempted size and reserve a
      // smaller bump instead. (max 3 times, chosen somewhat arbitrarily)
      for (var cutDown = 1; cutDown <= 4; cutDown *= 2) {
        var overGrownHeapSize = oldSize * (1 + 0.2 / cutDown); // ensure geometric growth
        // but limit overreserving (default to capping at +96MB overgrowth at most)
        overGrownHeapSize = Math.min(overGrownHeapSize, requestedSize + 100663296 );
  
        var newSize = Math.min(maxHeapSize, alignUp(Math.max(requestedSize, overGrownHeapSize), 65536));
  
        var replacement = emscripten_realloc_buffer(newSize);
        if (replacement) {
  
          return true;
        }
      }
      err('Failed to grow the heap from ' + oldSize + ' bytes to ' + newSize + ' bytes, not enough memory!');
      return false;
    }

  var ENV = {};
  
  function getExecutableName() {
      return thisProgram || './this.program';
    }
  function getEnvStrings() {
      if (!getEnvStrings.strings) {
        // Default values.
        // Browser language detection #8751
        var lang = ((typeof navigator == 'object' && navigator.languages && navigator.languages[0]) || 'C').replace('-', '_') + '.UTF-8';
        var env = {
          'USER': 'web_user',
          'LOGNAME': 'web_user',
          'PATH': '/',
          'PWD': '/',
          'HOME': '/home/web_user',
          'LANG': lang,
          '_': getExecutableName()
        };
        // Apply the user-provided values, if any.
        for (var x in ENV) {
          // x is a key in ENV; if ENV[x] is undefined, that means it was
          // explicitly set to be so. We allow user code to do that to
          // force variables with default values to remain unset.
          if (ENV[x] === undefined) delete env[x];
          else env[x] = ENV[x];
        }
        var strings = [];
        for (var x in env) {
          strings.push(x + '=' + env[x]);
        }
        getEnvStrings.strings = strings;
      }
      return getEnvStrings.strings;
    }
  
  function stringToAscii(str, buffer) {
      for (var i = 0; i < str.length; ++i) {
        assert(str.charCodeAt(i) === (str.charCodeAt(i) & 0xff));
        HEAP8[((buffer++)>>0)] = str.charCodeAt(i);
      }
      // Null-terminate the string
      HEAP8[((buffer)>>0)] = 0;
    }
  
  function _environ_get(__environ, environ_buf) {
      var bufSize = 0;
      getEnvStrings().forEach(function(string, i) {
        var ptr = environ_buf + bufSize;
        HEAPU32[(((__environ)+(i*4))>>2)] = ptr;
        stringToAscii(string, ptr);
        bufSize += string.length + 1;
      });
      return 0;
    }

  
  function _environ_sizes_get(penviron_count, penviron_buf_size) {
      var strings = getEnvStrings();
      HEAPU32[((penviron_count)>>2)] = strings.length;
      var bufSize = 0;
      strings.forEach(function(string) {
        bufSize += string.length + 1;
      });
      HEAPU32[((penviron_buf_size)>>2)] = bufSize;
      return 0;
    }

  function _fd_close(fd) {
  try {
  
      var stream = SYSCALLS.getStreamFromFD(fd);
      FS.close(stream);
      return 0;
    } catch (e) {
    if (typeof FS == 'undefined' || !(e.name === 'ErrnoError')) throw e;
    return e.errno;
  }
  }

  /** @param {number=} offset */
  function doReadv(stream, iov, iovcnt, offset) {
      var ret = 0;
      for (var i = 0; i < iovcnt; i++) {
        var ptr = HEAPU32[((iov)>>2)];
        var len = HEAPU32[(((iov)+(4))>>2)];
        iov += 8;
        var curr = FS.read(stream, HEAP8,ptr, len, offset);
        if (curr < 0) return -1;
        ret += curr;
        if (curr < len) break; // nothing more to read
        if (typeof offset !== 'undefined') {
          offset += curr;
        }
      }
      return ret;
    }
  
  function _fd_read(fd, iov, iovcnt, pnum) {
  try {
  
      var stream = SYSCALLS.getStreamFromFD(fd);
      var num = doReadv(stream, iov, iovcnt);
      HEAPU32[((pnum)>>2)] = num;
      return 0;
    } catch (e) {
    if (typeof FS == 'undefined' || !(e.name === 'ErrnoError')) throw e;
    return e.errno;
  }
  }

  function convertI32PairToI53Checked(lo, hi) {
      assert(lo == (lo >>> 0) || lo == (lo|0)); // lo should either be a i32 or a u32
      assert(hi === (hi|0));                    // hi should be a i32
      return ((hi + 0x200000) >>> 0 < 0x400001 - !!lo) ? (lo >>> 0) + hi * 4294967296 : NaN;
    }
  
  
  
  
  function _fd_seek(fd, offset_low, offset_high, whence, newOffset) {
  try {
  
      var offset = convertI32PairToI53Checked(offset_low, offset_high); if (isNaN(offset)) return 61;
      var stream = SYSCALLS.getStreamFromFD(fd);
      FS.llseek(stream, offset, whence);
      (tempI64 = [stream.position>>>0,(tempDouble=stream.position,(+(Math.abs(tempDouble))) >= 1.0 ? (tempDouble > 0.0 ? ((Math.min((+(Math.floor((tempDouble)/4294967296.0))), 4294967295.0))|0)>>>0 : (~~((+(Math.ceil((tempDouble - +(((~~(tempDouble)))>>>0))/4294967296.0)))))>>>0) : 0)],HEAP32[((newOffset)>>2)] = tempI64[0],HEAP32[(((newOffset)+(4))>>2)] = tempI64[1]);
      if (stream.getdents && offset === 0 && whence === 0) stream.getdents = null; // reset readdir state
      return 0;
    } catch (e) {
    if (typeof FS == 'undefined' || !(e.name === 'ErrnoError')) throw e;
    return e.errno;
  }
  }

  /** @param {number=} offset */
  function doWritev(stream, iov, iovcnt, offset) {
      var ret = 0;
      for (var i = 0; i < iovcnt; i++) {
        var ptr = HEAPU32[((iov)>>2)];
        var len = HEAPU32[(((iov)+(4))>>2)];
        iov += 8;
        var curr = FS.write(stream, HEAP8,ptr, len, offset);
        if (curr < 0) return -1;
        ret += curr;
        if (typeof offset !== 'undefined') {
          offset += curr;
        }
      }
      return ret;
    }
  
  function _fd_write(fd, iov, iovcnt, pnum) {
  try {
  
      var stream = SYSCALLS.getStreamFromFD(fd);
      var num = doWritev(stream, iov, iovcnt);
      HEAPU32[((pnum)>>2)] = num;
      return 0;
    } catch (e) {
    if (typeof FS == 'undefined' || !(e.name === 'ErrnoError')) throw e;
    return e.errno;
  }
  }

  function isLeapYear(year) {
        return year%4 === 0 && (year%100 !== 0 || year%400 === 0);
    }
  
  function arraySum(array, index) {
      var sum = 0;
      for (var i = 0; i <= index; sum += array[i++]) {
        // no-op
      }
      return sum;
    }
  
  
  var MONTH_DAYS_LEAP = [31,29,31,30,31,30,31,31,30,31,30,31];
  
  var MONTH_DAYS_REGULAR = [31,28,31,30,31,30,31,31,30,31,30,31];
  function addDays(date, days) {
      var newDate = new Date(date.getTime());
      while (days > 0) {
        var leap = isLeapYear(newDate.getFullYear());
        var currentMonth = newDate.getMonth();
        var daysInCurrentMonth = (leap ? MONTH_DAYS_LEAP : MONTH_DAYS_REGULAR)[currentMonth];
  
        if (days > daysInCurrentMonth-newDate.getDate()) {
          // we spill over to next month
          days -= (daysInCurrentMonth-newDate.getDate()+1);
          newDate.setDate(1);
          if (currentMonth < 11) {
            newDate.setMonth(currentMonth+1);
          } else {
            newDate.setMonth(0);
            newDate.setFullYear(newDate.getFullYear()+1);
          }
        } else {
          // we stay in current month
          newDate.setDate(newDate.getDate()+days);
          return newDate;
        }
      }
  
      return newDate;
    }
  
  
  
  
  function writeArrayToMemory(array, buffer) {
      assert(array.length >= 0, 'writeArrayToMemory array must have a length (should be an array or typed array)');
      HEAP8.set(array, buffer);
    }
  
  function _strftime(s, maxsize, format, tm) {
      // size_t strftime(char *restrict s, size_t maxsize, const char *restrict format, const struct tm *restrict timeptr);
      // http://pubs.opengroup.org/onlinepubs/009695399/functions/strftime.html
  
      var tm_zone = HEAP32[(((tm)+(40))>>2)];
  
      var date = {
        tm_sec: HEAP32[((tm)>>2)],
        tm_min: HEAP32[(((tm)+(4))>>2)],
        tm_hour: HEAP32[(((tm)+(8))>>2)],
        tm_mday: HEAP32[(((tm)+(12))>>2)],
        tm_mon: HEAP32[(((tm)+(16))>>2)],
        tm_year: HEAP32[(((tm)+(20))>>2)],
        tm_wday: HEAP32[(((tm)+(24))>>2)],
        tm_yday: HEAP32[(((tm)+(28))>>2)],
        tm_isdst: HEAP32[(((tm)+(32))>>2)],
        tm_gmtoff: HEAP32[(((tm)+(36))>>2)],
        tm_zone: tm_zone ? UTF8ToString(tm_zone) : ''
      };
  
      var pattern = UTF8ToString(format);
  
      // expand format
      var EXPANSION_RULES_1 = {
        '%c': '%a %b %d %H:%M:%S %Y',     // Replaced by the locale's appropriate date and time representation - e.g., Mon Aug  3 14:02:01 2013
        '%D': '%m/%d/%y',                 // Equivalent to %m / %d / %y
        '%F': '%Y-%m-%d',                 // Equivalent to %Y - %m - %d
        '%h': '%b',                       // Equivalent to %b
        '%r': '%I:%M:%S %p',              // Replaced by the time in a.m. and p.m. notation
        '%R': '%H:%M',                    // Replaced by the time in 24-hour notation
        '%T': '%H:%M:%S',                 // Replaced by the time
        '%x': '%m/%d/%y',                 // Replaced by the locale's appropriate date representation
        '%X': '%H:%M:%S',                 // Replaced by the locale's appropriate time representation
        // Modified Conversion Specifiers
        '%Ec': '%c',                      // Replaced by the locale's alternative appropriate date and time representation.
        '%EC': '%C',                      // Replaced by the name of the base year (period) in the locale's alternative representation.
        '%Ex': '%m/%d/%y',                // Replaced by the locale's alternative date representation.
        '%EX': '%H:%M:%S',                // Replaced by the locale's alternative time representation.
        '%Ey': '%y',                      // Replaced by the offset from %EC (year only) in the locale's alternative representation.
        '%EY': '%Y',                      // Replaced by the full alternative year representation.
        '%Od': '%d',                      // Replaced by the day of the month, using the locale's alternative numeric symbols, filled as needed with leading zeros if there is any alternative symbol for zero; otherwise, with leading <space> characters.
        '%Oe': '%e',                      // Replaced by the day of the month, using the locale's alternative numeric symbols, filled as needed with leading <space> characters.
        '%OH': '%H',                      // Replaced by the hour (24-hour clock) using the locale's alternative numeric symbols.
        '%OI': '%I',                      // Replaced by the hour (12-hour clock) using the locale's alternative numeric symbols.
        '%Om': '%m',                      // Replaced by the month using the locale's alternative numeric symbols.
        '%OM': '%M',                      // Replaced by the minutes using the locale's alternative numeric symbols.
        '%OS': '%S',                      // Replaced by the seconds using the locale's alternative numeric symbols.
        '%Ou': '%u',                      // Replaced by the weekday as a number in the locale's alternative representation (Monday=1).
        '%OU': '%U',                      // Replaced by the week number of the year (Sunday as the first day of the week, rules corresponding to %U ) using the locale's alternative numeric symbols.
        '%OV': '%V',                      // Replaced by the week number of the year (Monday as the first day of the week, rules corresponding to %V ) using the locale's alternative numeric symbols.
        '%Ow': '%w',                      // Replaced by the number of the weekday (Sunday=0) using the locale's alternative numeric symbols.
        '%OW': '%W',                      // Replaced by the week number of the year (Monday as the first day of the week) using the locale's alternative numeric symbols.
        '%Oy': '%y',                      // Replaced by the year (offset from %C ) using the locale's alternative numeric symbols.
      };
      for (var rule in EXPANSION_RULES_1) {
        pattern = pattern.replace(new RegExp(rule, 'g'), EXPANSION_RULES_1[rule]);
      }
  
      var WEEKDAYS = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
      var MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
  
      function leadingSomething(value, digits, character) {
        var str = typeof value == 'number' ? value.toString() : (value || '');
        while (str.length < digits) {
          str = character[0]+str;
        }
        return str;
      }
  
      function leadingNulls(value, digits) {
        return leadingSomething(value, digits, '0');
      }
  
      function compareByDay(date1, date2) {
        function sgn(value) {
          return value < 0 ? -1 : (value > 0 ? 1 : 0);
        }
  
        var compare;
        if ((compare = sgn(date1.getFullYear()-date2.getFullYear())) === 0) {
          if ((compare = sgn(date1.getMonth()-date2.getMonth())) === 0) {
            compare = sgn(date1.getDate()-date2.getDate());
          }
        }
        return compare;
      }
  
      function getFirstWeekStartDate(janFourth) {
          switch (janFourth.getDay()) {
            case 0: // Sunday
              return new Date(janFourth.getFullYear()-1, 11, 29);
            case 1: // Monday
              return janFourth;
            case 2: // Tuesday
              return new Date(janFourth.getFullYear(), 0, 3);
            case 3: // Wednesday
              return new Date(janFourth.getFullYear(), 0, 2);
            case 4: // Thursday
              return new Date(janFourth.getFullYear(), 0, 1);
            case 5: // Friday
              return new Date(janFourth.getFullYear()-1, 11, 31);
            case 6: // Saturday
              return new Date(janFourth.getFullYear()-1, 11, 30);
          }
      }
  
      function getWeekBasedYear(date) {
          var thisDate = addDays(new Date(date.tm_year+1900, 0, 1), date.tm_yday);
  
          var janFourthThisYear = new Date(thisDate.getFullYear(), 0, 4);
          var janFourthNextYear = new Date(thisDate.getFullYear()+1, 0, 4);
  
          var firstWeekStartThisYear = getFirstWeekStartDate(janFourthThisYear);
          var firstWeekStartNextYear = getFirstWeekStartDate(janFourthNextYear);
  
          if (compareByDay(firstWeekStartThisYear, thisDate) <= 0) {
            // this date is after the start of the first week of this year
            if (compareByDay(firstWeekStartNextYear, thisDate) <= 0) {
              return thisDate.getFullYear()+1;
            }
            return thisDate.getFullYear();
          }
          return thisDate.getFullYear()-1;
      }
  
      var EXPANSION_RULES_2 = {
        '%a': function(date) {
          return WEEKDAYS[date.tm_wday].substring(0,3);
        },
        '%A': function(date) {
          return WEEKDAYS[date.tm_wday];
        },
        '%b': function(date) {
          return MONTHS[date.tm_mon].substring(0,3);
        },
        '%B': function(date) {
          return MONTHS[date.tm_mon];
        },
        '%C': function(date) {
          var year = date.tm_year+1900;
          return leadingNulls((year/100)|0,2);
        },
        '%d': function(date) {
          return leadingNulls(date.tm_mday, 2);
        },
        '%e': function(date) {
          return leadingSomething(date.tm_mday, 2, ' ');
        },
        '%g': function(date) {
          // %g, %G, and %V give values according to the ISO 8601:2000 standard week-based year.
          // In this system, weeks begin on a Monday and week 1 of the year is the week that includes
          // January 4th, which is also the week that includes the first Thursday of the year, and
          // is also the first week that contains at least four days in the year.
          // If the first Monday of January is the 2nd, 3rd, or 4th, the preceding days are part of
          // the last week of the preceding year; thus, for Saturday 2nd January 1999,
          // %G is replaced by 1998 and %V is replaced by 53. If December 29th, 30th,
          // or 31st is a Monday, it and any following days are part of week 1 of the following year.
          // Thus, for Tuesday 30th December 1997, %G is replaced by 1998 and %V is replaced by 01.
  
          return getWeekBasedYear(date).toString().substring(2);
        },
        '%G': function(date) {
          return getWeekBasedYear(date);
        },
        '%H': function(date) {
          return leadingNulls(date.tm_hour, 2);
        },
        '%I': function(date) {
          var twelveHour = date.tm_hour;
          if (twelveHour == 0) twelveHour = 12;
          else if (twelveHour > 12) twelveHour -= 12;
          return leadingNulls(twelveHour, 2);
        },
        '%j': function(date) {
          // Day of the year (001-366)
          return leadingNulls(date.tm_mday + arraySum(isLeapYear(date.tm_year+1900) ? MONTH_DAYS_LEAP : MONTH_DAYS_REGULAR, date.tm_mon-1), 3);
        },
        '%m': function(date) {
          return leadingNulls(date.tm_mon+1, 2);
        },
        '%M': function(date) {
          return leadingNulls(date.tm_min, 2);
        },
        '%n': function() {
          return '\n';
        },
        '%p': function(date) {
          if (date.tm_hour >= 0 && date.tm_hour < 12) {
            return 'AM';
          }
          return 'PM';
        },
        '%S': function(date) {
          return leadingNulls(date.tm_sec, 2);
        },
        '%t': function() {
          return '\t';
        },
        '%u': function(date) {
          return date.tm_wday || 7;
        },
        '%U': function(date) {
          var days = date.tm_yday + 7 - date.tm_wday;
          return leadingNulls(Math.floor(days / 7), 2);
        },
        '%V': function(date) {
          // Replaced by the week number of the year (Monday as the first day of the week)
          // as a decimal number [01,53]. If the week containing 1 January has four
          // or more days in the new year, then it is considered week 1.
          // Otherwise, it is the last week of the previous year, and the next week is week 1.
          // Both January 4th and the first Thursday of January are always in week 1. [ tm_year, tm_wday, tm_yday]
          var val = Math.floor((date.tm_yday + 7 - (date.tm_wday + 6) % 7 ) / 7);
          // If 1 Jan is just 1-3 days past Monday, the previous week
          // is also in this year.
          if ((date.tm_wday + 371 - date.tm_yday - 2) % 7 <= 2) {
            val++;
          }
          if (!val) {
            val = 52;
            // If 31 December of prev year a Thursday, or Friday of a
            // leap year, then the prev year has 53 weeks.
            var dec31 = (date.tm_wday + 7 - date.tm_yday - 1) % 7;
            if (dec31 == 4 || (dec31 == 5 && isLeapYear(date.tm_year%400-1))) {
              val++;
            }
          } else if (val == 53) {
            // If 1 January is not a Thursday, and not a Wednesday of a
            // leap year, then this year has only 52 weeks.
            var jan1 = (date.tm_wday + 371 - date.tm_yday) % 7;
            if (jan1 != 4 && (jan1 != 3 || !isLeapYear(date.tm_year)))
              val = 1;
          }
          return leadingNulls(val, 2);
        },
        '%w': function(date) {
          return date.tm_wday;
        },
        '%W': function(date) {
          var days = date.tm_yday + 7 - ((date.tm_wday + 6) % 7);
          return leadingNulls(Math.floor(days / 7), 2);
        },
        '%y': function(date) {
          // Replaced by the last two digits of the year as a decimal number [00,99]. [ tm_year]
          return (date.tm_year+1900).toString().substring(2);
        },
        '%Y': function(date) {
          // Replaced by the year as a decimal number (for example, 1997). [ tm_year]
          return date.tm_year+1900;
        },
        '%z': function(date) {
          // Replaced by the offset from UTC in the ISO 8601:2000 standard format ( +hhmm or -hhmm ).
          // For example, "-0430" means 4 hours 30 minutes behind UTC (west of Greenwich).
          var off = date.tm_gmtoff;
          var ahead = off >= 0;
          off = Math.abs(off) / 60;
          // convert from minutes into hhmm format (which means 60 minutes = 100 units)
          off = (off / 60)*100 + (off % 60);
          return (ahead ? '+' : '-') + String("0000" + off).slice(-4);
        },
        '%Z': function(date) {
          return date.tm_zone;
        },
        '%%': function() {
          return '%';
        }
      };
  
      // Replace %% with a pair of NULLs (which cannot occur in a C string), then
      // re-inject them after processing.
      pattern = pattern.replace(/%%/g, '\0\0');
      for (var rule in EXPANSION_RULES_2) {
        if (pattern.includes(rule)) {
          pattern = pattern.replace(new RegExp(rule, 'g'), EXPANSION_RULES_2[rule](date));
        }
      }
      pattern = pattern.replace(/\0\0/g, '%');
  
      var bytes = intArrayFromString(pattern, false);
      if (bytes.length > maxsize) {
        return 0;
      }
  
      writeArrayToMemory(bytes, s);
      return bytes.length-1;
    }
  function _strftime_l(s, maxsize, format, tm, loc) {
      return _strftime(s, maxsize, format, tm); // no locale support yet
    }


  var FSNode = /** @constructor */ function(parent, name, mode, rdev) {
    if (!parent) {
      parent = this;  // root node sets parent to itself
    }
    this.parent = parent;
    this.mount = parent.mount;
    this.mounted = null;
    this.id = FS.nextInode++;
    this.name = name;
    this.mode = mode;
    this.node_ops = {};
    this.stream_ops = {};
    this.rdev = rdev;
  };
  var readMode = 292/*292*/ | 73/*73*/;
  var writeMode = 146/*146*/;
  Object.defineProperties(FSNode.prototype, {
   read: {
    get: /** @this{FSNode} */function() {
     return (this.mode & readMode) === readMode;
    },
    set: /** @this{FSNode} */function(val) {
     val ? this.mode |= readMode : this.mode &= ~readMode;
    }
   },
   write: {
    get: /** @this{FSNode} */function() {
     return (this.mode & writeMode) === writeMode;
    },
    set: /** @this{FSNode} */function(val) {
     val ? this.mode |= writeMode : this.mode &= ~writeMode;
    }
   },
   isFolder: {
    get: /** @this{FSNode} */function() {
     return FS.isDir(this.mode);
    }
   },
   isDevice: {
    get: /** @this{FSNode} */function() {
     return FS.isChrdev(this.mode);
    }
   }
  });
  FS.FSNode = FSNode;
  FS.staticInit();ERRNO_CODES = {
      'EPERM': 63,
      'ENOENT': 44,
      'ESRCH': 71,
      'EINTR': 27,
      'EIO': 29,
      'ENXIO': 60,
      'E2BIG': 1,
      'ENOEXEC': 45,
      'EBADF': 8,
      'ECHILD': 12,
      'EAGAIN': 6,
      'EWOULDBLOCK': 6,
      'ENOMEM': 48,
      'EACCES': 2,
      'EFAULT': 21,
      'ENOTBLK': 105,
      'EBUSY': 10,
      'EEXIST': 20,
      'EXDEV': 75,
      'ENODEV': 43,
      'ENOTDIR': 54,
      'EISDIR': 31,
      'EINVAL': 28,
      'ENFILE': 41,
      'EMFILE': 33,
      'ENOTTY': 59,
      'ETXTBSY': 74,
      'EFBIG': 22,
      'ENOSPC': 51,
      'ESPIPE': 70,
      'EROFS': 69,
      'EMLINK': 34,
      'EPIPE': 64,
      'EDOM': 18,
      'ERANGE': 68,
      'ENOMSG': 49,
      'EIDRM': 24,
      'ECHRNG': 106,
      'EL2NSYNC': 156,
      'EL3HLT': 107,
      'EL3RST': 108,
      'ELNRNG': 109,
      'EUNATCH': 110,
      'ENOCSI': 111,
      'EL2HLT': 112,
      'EDEADLK': 16,
      'ENOLCK': 46,
      'EBADE': 113,
      'EBADR': 114,
      'EXFULL': 115,
      'ENOANO': 104,
      'EBADRQC': 103,
      'EBADSLT': 102,
      'EDEADLOCK': 16,
      'EBFONT': 101,
      'ENOSTR': 100,
      'ENODATA': 116,
      'ETIME': 117,
      'ENOSR': 118,
      'ENONET': 119,
      'ENOPKG': 120,
      'EREMOTE': 121,
      'ENOLINK': 47,
      'EADV': 122,
      'ESRMNT': 123,
      'ECOMM': 124,
      'EPROTO': 65,
      'EMULTIHOP': 36,
      'EDOTDOT': 125,
      'EBADMSG': 9,
      'ENOTUNIQ': 126,
      'EBADFD': 127,
      'EREMCHG': 128,
      'ELIBACC': 129,
      'ELIBBAD': 130,
      'ELIBSCN': 131,
      'ELIBMAX': 132,
      'ELIBEXEC': 133,
      'ENOSYS': 52,
      'ENOTEMPTY': 55,
      'ENAMETOOLONG': 37,
      'ELOOP': 32,
      'EOPNOTSUPP': 138,
      'EPFNOSUPPORT': 139,
      'ECONNRESET': 15,
      'ENOBUFS': 42,
      'EAFNOSUPPORT': 5,
      'EPROTOTYPE': 67,
      'ENOTSOCK': 57,
      'ENOPROTOOPT': 50,
      'ESHUTDOWN': 140,
      'ECONNREFUSED': 14,
      'EADDRINUSE': 3,
      'ECONNABORTED': 13,
      'ENETUNREACH': 40,
      'ENETDOWN': 38,
      'ETIMEDOUT': 73,
      'EHOSTDOWN': 142,
      'EHOSTUNREACH': 23,
      'EINPROGRESS': 26,
      'EALREADY': 7,
      'EDESTADDRREQ': 17,
      'EMSGSIZE': 35,
      'EPROTONOSUPPORT': 66,
      'ESOCKTNOSUPPORT': 137,
      'EADDRNOTAVAIL': 4,
      'ENETRESET': 39,
      'EISCONN': 30,
      'ENOTCONN': 53,
      'ETOOMANYREFS': 141,
      'EUSERS': 136,
      'EDQUOT': 19,
      'ESTALE': 72,
      'ENOTSUP': 138,
      'ENOMEDIUM': 148,
      'EILSEQ': 25,
      'EOVERFLOW': 61,
      'ECANCELED': 11,
      'ENOTRECOVERABLE': 56,
      'EOWNERDEAD': 62,
      'ESTRPIPE': 135,
    };embind_init_charCodes();
BindingError = Module['BindingError'] = extendError(Error, 'BindingError');InternalError = Module['InternalError'] = extendError(Error, 'InternalError');init_ClassHandle();
init_embind();init_RegisteredPointer();
UnboundTypeError = Module['UnboundTypeError'] = extendError(Error, 'UnboundTypeError');init_emval();// include: base64Utils.js
// Copied from https://github.com/strophe/strophejs/blob/e06d027/src/polyfills.js#L149

// This code was written by Tyler Akins and has been placed in the
// public domain.  It would be nice if you left this header intact.
// Base64 code from Tyler Akins -- http://rumkin.com

/**
 * Decodes a base64 string.
 * @param {string} input The string to decode.
 */
var decodeBase64 = typeof atob == 'function' ? atob : function (input) {
  var keyStr = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=';

  var output = '';
  var chr1, chr2, chr3;
  var enc1, enc2, enc3, enc4;
  var i = 0;
  // remove all characters that are not A-Z, a-z, 0-9, +, /, or =
  input = input.replace(/[^A-Za-z0-9\+\/\=]/g, '');
  do {
    enc1 = keyStr.indexOf(input.charAt(i++));
    enc2 = keyStr.indexOf(input.charAt(i++));
    enc3 = keyStr.indexOf(input.charAt(i++));
    enc4 = keyStr.indexOf(input.charAt(i++));

    chr1 = (enc1 << 2) | (enc2 >> 4);
    chr2 = ((enc2 & 15) << 4) | (enc3 >> 2);
    chr3 = ((enc3 & 3) << 6) | enc4;

    output = output + String.fromCharCode(chr1);

    if (enc3 !== 64) {
      output = output + String.fromCharCode(chr2);
    }
    if (enc4 !== 64) {
      output = output + String.fromCharCode(chr3);
    }
  } while (i < input.length);
  return output;
};

// Converts a string of base64 into a byte array.
// Throws error on invalid input.
function intArrayFromBase64(s) {

  try {
    var decoded = decodeBase64(s);
    var bytes = new Uint8Array(decoded.length);
    for (var i = 0 ; i < decoded.length ; ++i) {
      bytes[i] = decoded.charCodeAt(i);
    }
    return bytes;
  } catch (_) {
    throw new Error('Converting base64 string to bytes failed.');
  }
}

// If filename is a base64 data URI, parses and returns data (Buffer on node,
// Uint8Array otherwise). If filename is not a base64 data URI, returns undefined.
function tryParseAsDataURI(filename) {
  if (!isDataURI(filename)) {
    return;
  }

  return intArrayFromBase64(filename.slice(dataURIPrefix.length));
}


// end include: base64Utils.js
function checkIncomingModuleAPI() {
  ignoredModuleProp('fetchSettings');
}
var wasmImports = {
  "__assert_fail": ___assert_fail,
  "__syscall_fcntl64": ___syscall_fcntl64,
  "__syscall_fstat64": ___syscall_fstat64,
  "__syscall_ioctl": ___syscall_ioctl,
  "__syscall_lstat64": ___syscall_lstat64,
  "__syscall_newfstatat": ___syscall_newfstatat,
  "__syscall_openat": ___syscall_openat,
  "__syscall_stat64": ___syscall_stat64,
  "__throw_exception_with_stack_trace": ___throw_exception_with_stack_trace,
  "_embind_register_bigint": __embind_register_bigint,
  "_embind_register_bool": __embind_register_bool,
  "_embind_register_class": __embind_register_class,
  "_embind_register_class_class_function": __embind_register_class_class_function,
  "_embind_register_class_constructor": __embind_register_class_constructor,
  "_embind_register_class_function": __embind_register_class_function,
  "_embind_register_emval": __embind_register_emval,
  "_embind_register_float": __embind_register_float,
  "_embind_register_function": __embind_register_function,
  "_embind_register_integer": __embind_register_integer,
  "_embind_register_memory_view": __embind_register_memory_view,
  "_embind_register_std_string": __embind_register_std_string,
  "_embind_register_std_wstring": __embind_register_std_wstring,
  "_embind_register_void": __embind_register_void,
  "_emval_as": __emval_as,
  "_emval_call_method": __emval_call_method,
  "_emval_call_void_method": __emval_call_void_method,
  "_emval_decref": __emval_decref,
  "_emval_get_method_caller": __emval_get_method_caller,
  "_emval_get_property": __emval_get_property,
  "_emval_incref": __emval_incref,
  "_emval_new_array": __emval_new_array,
  "_emval_new_array_from_memory_view": __emval_new_array_from_memory_view,
  "_emval_new_cstring": __emval_new_cstring,
  "_emval_new_object": __emval_new_object,
  "_emval_run_destructors": __emval_run_destructors,
  "_emval_set_property": __emval_set_property,
  "_emval_take_value": __emval_take_value,
  "abort": _abort,
  "emscripten_asm_const_int": _emscripten_asm_const_int,
  "emscripten_memcpy_big": _emscripten_memcpy_big,
  "emscripten_resize_heap": _emscripten_resize_heap,
  "environ_get": _environ_get,
  "environ_sizes_get": _environ_sizes_get,
  "fd_close": _fd_close,
  "fd_read": _fd_read,
  "fd_seek": _fd_seek,
  "fd_write": _fd_write,
  "strftime_l": _strftime_l,
  "syncIdb_js": syncIdb_js
};
createWasm();
/** @type {function(...*):?} */
var _malloc = createExportWrapper("malloc");
/** @type {function(...*):?} */
var _free = Module["_free"] = createExportWrapper("free");
/** @type {function(...*):?} */
var ___getTypeName = Module["___getTypeName"] = createExportWrapper("__getTypeName");
/** @type {function(...*):?} */
Module["__embind_initialize_bindings"] = createExportWrapper("_embind_initialize_bindings");
/** @type {function(...*):?} */
var ___errno_location = createExportWrapper("__errno_location");
/** @type {function(...*):?} */
var _fflush = Module["_fflush"] = createExportWrapper("fflush");
/** @type {function(...*):?} */
var ___trap = function() {
  return (___trap = Module["asm"]["__trap"]).apply(null, arguments);
};

/** @type {function(...*):?} */
var _emscripten_stack_init = function() {
  return (_emscripten_stack_init = Module["asm"]["emscripten_stack_init"]).apply(null, arguments);
};

/** @type {function(...*):?} */
var _emscripten_stack_get_end = function() {
  return (_emscripten_stack_get_end = Module["asm"]["emscripten_stack_get_end"]).apply(null, arguments);
};

/** @type {function(...*):?} */
var stackSave = createExportWrapper("stackSave");
/** @type {function(...*):?} */
var stackRestore = createExportWrapper("stackRestore");
/** @type {function(...*):?} */
var stackAlloc = createExportWrapper("stackAlloc");

/** @type {function(...*):?} */
Module["___cxa_decrement_exception_refcount"] = createExportWrapper("__cxa_decrement_exception_refcount");
/** @type {function(...*):?} */
Module["___cxa_increment_exception_refcount"] = createExportWrapper("__cxa_increment_exception_refcount");
/** @type {function(...*):?} */
var ___cxa_demangle = createExportWrapper("__cxa_demangle");
/** @type {function(...*):?} */
var ___thrown_object_from_unwind_exception = Module["___thrown_object_from_unwind_exception"] = createExportWrapper("__thrown_object_from_unwind_exception");
/** @type {function(...*):?} */
var ___get_exception_message = Module["___get_exception_message"] = createExportWrapper("__get_exception_message");
/** @type {function(...*):?} */
Module["dynCall_jiji"] = createExportWrapper("dynCall_jiji");
/** @type {function(...*):?} */
Module["dynCall_viijii"] = createExportWrapper("dynCall_viijii");
/** @type {function(...*):?} */
Module["dynCall_iiiiij"] = createExportWrapper("dynCall_iiiiij");
/** @type {function(...*):?} */
Module["dynCall_iiiiijj"] = createExportWrapper("dynCall_iiiiijj");
/** @type {function(...*):?} */
Module["dynCall_iiiiiijj"] = createExportWrapper("dynCall_iiiiiijj");
Module['___start_em_js'] = 44648;
Module['___stop_em_js'] = 44983;

// include: postamble.js
// === Auto-generated postamble setup entry stuff ===

var missingLibrarySymbols = [
  'exitJS',
  'ydayFromDate',
  'inetPton4',
  'inetNtop4',
  'inetPton6',
  'inetNtop6',
  'readSockaddr',
  'writeSockaddr',
  'getHostByName',
  'traverseStack',
  'getCallstack',
  'emscriptenLog',
  'convertPCtoSourceLocation',
  'runMainThreadEmAsm',
  'jstoi_q',
  'jstoi_s',
  'listenOnce',
  'autoResumeAudioContext',
  'handleException',
  'runtimeKeepalivePush',
  'runtimeKeepalivePop',
  'callUserCallback',
  'maybeExit',
  'safeSetTimeout',
  'asmjsMangle',
  'getNativeTypeSize',
  'STACK_SIZE',
  'STACK_ALIGN',
  'POINTER_SIZE',
  'ASSERTIONS',
  'writeI53ToI64',
  'writeI53ToI64Clamped',
  'writeI53ToI64Signaling',
  'writeI53ToU64Clamped',
  'writeI53ToU64Signaling',
  'readI53FromI64',
  'readI53FromU64',
  'convertI32PairToI53',
  'convertU32PairToI53',
  'getCFunc',
  'ccall',
  'cwrap',
  'uleb128Encode',
  'sigToWasmTypes',
  'generateFuncType',
  'convertJsFunctionToWasm',
  'getEmptyTableSlot',
  'updateTableMap',
  'getFunctionAddress',
  'addFunction',
  'removeFunction',
  'reallyNegative',
  'unSign',
  'strLen',
  'reSign',
  'formatString',
  'AsciiToString',
  'stringToNewUTF8',
  'getSocketFromFD',
  'getSocketAddress',
  'registerKeyEventCallback',
  'maybeCStringToJsString',
  'findEventTarget',
  'findCanvasEventTarget',
  'getBoundingClientRect',
  'fillMouseEventData',
  'registerMouseEventCallback',
  'registerWheelEventCallback',
  'registerUiEventCallback',
  'registerFocusEventCallback',
  'fillDeviceOrientationEventData',
  'registerDeviceOrientationEventCallback',
  'fillDeviceMotionEventData',
  'registerDeviceMotionEventCallback',
  'screenOrientation',
  'fillOrientationChangeEventData',
  'registerOrientationChangeEventCallback',
  'fillFullscreenChangeEventData',
  'registerFullscreenChangeEventCallback',
  'JSEvents_requestFullscreen',
  'JSEvents_resizeCanvasForFullscreen',
  'registerRestoreOldStyle',
  'hideEverythingExceptGivenElement',
  'restoreHiddenElements',
  'setLetterbox',
  'softFullscreenResizeWebGLRenderTarget',
  'doRequestFullscreen',
  'fillPointerlockChangeEventData',
  'registerPointerlockChangeEventCallback',
  'registerPointerlockErrorEventCallback',
  'requestPointerLock',
  'fillVisibilityChangeEventData',
  'registerVisibilityChangeEventCallback',
  'registerTouchEventCallback',
  'fillGamepadEventData',
  'registerGamepadEventCallback',
  'registerBeforeUnloadEventCallback',
  'fillBatteryEventData',
  'battery',
  'registerBatteryEventCallback',
  'setCanvasElementSize',
  'getCanvasElementSize',
  'checkWasiClock',
  'wasiRightsToMuslOFlags',
  'wasiOFlagsToMuslOFlags',
  'createDyncallWrapper',
  'setImmediateWrapped',
  'clearImmediateWrapped',
  'polyfillSetImmediate',
  'getPromise',
  'makePromise',
  'makePromiseCallback',
  'setMainLoop',
  '_setNetworkCallback',
  'heapObjectForWebGLType',
  'heapAccessShiftForWebGLHeap',
  'webgl_enable_ANGLE_instanced_arrays',
  'webgl_enable_OES_vertex_array_object',
  'webgl_enable_WEBGL_draw_buffers',
  'webgl_enable_WEBGL_multi_draw',
  'emscriptenWebGLGet',
  'computeUnpackAlignedImageSize',
  'colorChannelsInGlTextureFormat',
  'emscriptenWebGLGetTexPixelData',
  '__glGenObject',
  'emscriptenWebGLGetUniform',
  'webglGetUniformLocation',
  'webglPrepareUniformLocationsBeforeFirstUse',
  'webglGetLeftBracePos',
  'emscriptenWebGLGetVertexAttrib',
  '__glGetActiveAttribOrUniform',
  'writeGLArray',
  'registerWebGlEventCallback',
  'runAndAbortIfError',
  'SDL_unicode',
  'SDL_ttfContext',
  'SDL_audio',
  'GLFW_Window',
  'ALLOC_NORMAL',
  'ALLOC_STACK',
  'allocate',
  'writeStringToMemory',
  'writeAsciiToMemory',
  'registerInheritedInstance',
  'unregisterInheritedInstance',
  'enumReadValueFromPointer',
  'validateThis',
  'craftEmvalAllocator',
  'emval_get_global',
];
missingLibrarySymbols.forEach(missingLibrarySymbol);

var unexportedSymbols = [
  'run',
  'addOnPreRun',
  'addOnInit',
  'addOnPreMain',
  'addOnExit',
  'addOnPostRun',
  'addRunDependency',
  'removeRunDependency',
  'FS_createFolder',
  'FS_createPath',
  'FS_createDataFile',
  'FS_createPreloadedFile',
  'FS_createLazyFile',
  'FS_createLink',
  'FS_createDevice',
  'FS_unlink',
  'out',
  'err',
  'callMain',
  'abort',
  'keepRuntimeAlive',
  'wasmMemory',
  'stackAlloc',
  'stackSave',
  'stackRestore',
  'getTempRet0',
  'setTempRet0',
  'writeStackCookie',
  'checkStackCookie',
  'intArrayFromBase64',
  'tryParseAsDataURI',
  'ptrToString',
  'zeroMemory',
  'getHeapMax',
  'emscripten_realloc_buffer',
  'ENV',
  'MONTH_DAYS_REGULAR',
  'MONTH_DAYS_LEAP',
  'MONTH_DAYS_REGULAR_CUMULATIVE',
  'MONTH_DAYS_LEAP_CUMULATIVE',
  'isLeapYear',
  'arraySum',
  'addDays',
  'ERRNO_CODES',
  'ERRNO_MESSAGES',
  'setErrNo',
  'DNS',
  'Protocols',
  'Sockets',
  'initRandomFill',
  'randomFill',
  'timers',
  'warnOnce',
  'UNWIND_CACHE',
  'readEmAsmArgsArray',
  'readEmAsmArgs',
  'runEmAsmFunction',
  'getExecutableName',
  'dynCallLegacy',
  'getDynCaller',
  'dynCall',
  'asyncLoad',
  'alignMemory',
  'mmapAlloc',
  'HandleAllocator',
  'convertI32PairToI53Checked',
  'freeTableIndexes',
  'functionsInTableMap',
  'setValue',
  'getValue',
  'PATH',
  'PATH_FS',
  'UTF8Decoder',
  'UTF8ArrayToString',
  'UTF8ToString',
  'stringToUTF8Array',
  'stringToUTF8',
  'lengthBytesUTF8',
  'intArrayFromString',
  'intArrayToString',
  'stringToAscii',
  'UTF16Decoder',
  'UTF16ToString',
  'stringToUTF16',
  'lengthBytesUTF16',
  'UTF32ToString',
  'stringToUTF32',
  'lengthBytesUTF32',
  'stringToUTF8OnStack',
  'writeArrayToMemory',
  'SYSCALLS',
  'JSEvents',
  'specialHTMLTargets',
  'currentFullscreenStrategy',
  'restoreOldWindowedStyle',
  'demangle',
  'demangleAll',
  'jsStackTrace',
  'stackTrace',
  'ExitStatus',
  'getEnvStrings',
  'doReadv',
  'doWritev',
  'dlopenMissingError',
  'promiseMap',
  'getExceptionMessageCommon',
  'getCppExceptionTag',
  'getCppExceptionThrownObjectFromWebAssemblyException',
  'incrementExceptionRefcount',
  'decrementExceptionRefcount',
  'getExceptionMessage',
  'Browser',
  'wget',
  'FS',
  'MEMFS',
  'TTY',
  'PIPEFS',
  'SOCKFS',
  'tempFixedLengthArray',
  'miniTempWebGLFloatBuffers',
  'miniTempWebGLIntBuffers',
  'GL',
  'emscripten_webgl_power_preferences',
  'AL',
  'GLUT',
  'EGL',
  'GLEW',
  'IDBStore',
  'SDL',
  'SDL_gfx',
  'GLFW',
  'allocateUTF8',
  'allocateUTF8OnStack',
  'InternalError',
  'BindingError',
  'UnboundTypeError',
  'PureVirtualError',
  'init_embind',
  'throwInternalError',
  'throwBindingError',
  'throwUnboundTypeError',
  'ensureOverloadTable',
  'exposePublicSymbol',
  'replacePublicSymbol',
  'extendError',
  'createNamedFunction',
  'embindRepr',
  'registeredInstances',
  'getBasestPointer',
  'getInheritedInstance',
  'getInheritedInstanceCount',
  'getLiveInheritedInstances',
  'registeredTypes',
  'awaitingDependencies',
  'typeDependencies',
  'registeredPointers',
  'registerType',
  'whenDependentTypesAreResolved',
  'embind_charCodes',
  'embind_init_charCodes',
  'readLatin1String',
  'getTypeName',
  'heap32VectorToArray',
  'requireRegisteredType',
  'getShiftFromSize',
  'integerReadValueFromPointer',
  'floatReadValueFromPointer',
  'simpleReadValueFromPointer',
  'runDestructors',
  'newFunc',
  'craftInvokerFunction',
  'embind__requireFunction',
  'tupleRegistrations',
  'structRegistrations',
  'genericPointerToWireType',
  'constNoSmartPtrRawPointerToWireType',
  'nonConstNoSmartPtrRawPointerToWireType',
  'init_RegisteredPointer',
  'RegisteredPointer',
  'RegisteredPointer_getPointee',
  'RegisteredPointer_destructor',
  'RegisteredPointer_deleteObject',
  'RegisteredPointer_fromWireType',
  'runDestructor',
  'releaseClassHandle',
  'finalizationRegistry',
  'detachFinalizer_deps',
  'detachFinalizer',
  'attachFinalizer',
  'makeClassHandle',
  'init_ClassHandle',
  'ClassHandle',
  'ClassHandle_isAliasOf',
  'throwInstanceAlreadyDeleted',
  'ClassHandle_clone',
  'ClassHandle_delete',
  'deletionQueue',
  'ClassHandle_isDeleted',
  'ClassHandle_deleteLater',
  'flushPendingDeletes',
  'delayFunction',
  'setDelayFunction',
  'RegisteredClass',
  'shallowCopyInternalPointer',
  'downcastPointer',
  'upcastPointer',
  'char_0',
  'char_9',
  'makeLegalFunctionName',
  'emval_handles',
  'emval_symbols',
  'init_emval',
  'count_emval_handles',
  'getStringOrSymbol',
  'Emval',
  'emval_newers',
  'emval_lookupTypes',
  'emval_allocateDestructors',
  'emval_methodCallers',
  'emval_addMethodCaller',
  'emval_registeredMethods',
  'IDBFS',
];
unexportedSymbols.forEach(unexportedRuntimeSymbol);



var calledRun;

dependenciesFulfilled = function runCaller() {
  // If run has never been called, and we should call run (INVOKE_RUN is true, and Module.noInitialRun is not false)
  if (!calledRun) run();
  if (!calledRun) dependenciesFulfilled = runCaller; // try this again later, after new deps are fulfilled
};

function stackCheckInit() {
  // This is normally called automatically during __wasm_call_ctors but need to
  // get these values before even running any of the ctors so we call it redundantly
  // here.
  _emscripten_stack_init();
  // TODO(sbc): Move writeStackCookie to native to to avoid this.
  writeStackCookie();
}

function run() {

  if (runDependencies > 0) {
    return;
  }

    stackCheckInit();

  preRun();

  // a preRun added a dependency, run will be called later
  if (runDependencies > 0) {
    return;
  }

  function doRun() {
    // run may have just been called through dependencies being fulfilled just in this very frame,
    // or while the async setStatus time below was happening
    if (calledRun) return;
    calledRun = true;
    Module['calledRun'] = true;

    if (ABORT) return;

    initRuntime();

    readyPromiseResolve(Module);
    if (Module['onRuntimeInitialized']) Module['onRuntimeInitialized']();

    assert(!Module['_main'], 'compiled without a main, but one is present. if you added it from JS, use Module["onRuntimeInitialized"]');

    postRun();
  }

  if (Module['setStatus']) {
    Module['setStatus']('Running...');
    setTimeout(function() {
      setTimeout(function() {
        Module['setStatus']('');
      }, 1);
      doRun();
    }, 1);
  } else
  {
    doRun();
  }
  checkStackCookie();
}

if (Module['preInit']) {
  if (typeof Module['preInit'] == 'function') Module['preInit'] = [Module['preInit']];
  while (Module['preInit'].length > 0) {
    Module['preInit'].pop()();
  }
}

run();


// end include: postamble.js


  return hnswlib.ready
}

);
})();

export { hnswlib as default };
//# sourceMappingURL=hnswlib-317962d7.js.map
