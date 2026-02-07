let IDBFS_STORE_NAME, defaultParams, hnswParamsForAda, loadHnswlib, syncFileSystem, waitForFileSystemInitalized, waitForFileSystemSynced;
let __tla = (async () => {
  defaultParams = {
    initIndex: [
      32,
      128,
      100
    ]
  };
  hnswParamsForAda = {
    m: 32,
    efSearch: 128,
    efConstruction: 128,
    numNeighbors: 8,
    dimensions: 1538
  };
  IDBFS_STORE_NAME = "FILE_DATA";
  let library;
  syncFileSystem = (action) => {
    const EmscriptenFileSystemManager = library.EmscriptenFileSystemManager;
    const syncAction = action === "read" ? true : action === "write" ? false : void 0;
    if (syncAction === void 0)
      throw new Error("Invalid action type");
    return new Promise((resolve, reject) => {
      try {
        EmscriptenFileSystemManager.syncFS(syncAction, () => {
          resolve();
        });
      } catch (error) {
        reject(error);
      }
    });
  };
  waitForFileSystemInitalized = () => {
    const EmscriptenFileSystemManager = library.EmscriptenFileSystemManager;
    return new Promise((resolve, reject) => {
      let totalWaitTime = 0;
      const checkInterval = 100;
      const maxWaitTime = 4e3;
      const checkInitialization = () => {
        if (EmscriptenFileSystemManager.isInitialized()) {
          resolve();
        } else if (totalWaitTime >= maxWaitTime) {
          reject(new Error("Failed to initialize filesystem"));
        } else {
          totalWaitTime += checkInterval;
          setTimeout(checkInitialization, checkInterval);
        }
      };
      setTimeout(checkInitialization, checkInterval);
    });
  };
  waitForFileSystemSynced = () => {
    const EmscriptenFileSystemManager = library.EmscriptenFileSystemManager;
    return new Promise((resolve, reject) => {
      let totalWaitTime = 0;
      const checkInterval = 100;
      const maxWaitTime = 4e3;
      const checkInitialization = () => {
        if (EmscriptenFileSystemManager.isSynced()) {
          resolve();
        } else if (totalWaitTime >= maxWaitTime) {
          reject(new Error("Failed to initialize filesystem"));
        } else {
          totalWaitTime += checkInterval;
          setTimeout(checkInitialization, checkInterval);
        }
      };
      setTimeout(checkInitialization, checkInterval);
    });
  };
  const initializeFileSystemAsync = async (inputFsType) => {
    const fsType = inputFsType == null ? "IDBFS" : inputFsType;
    const EmscriptenFileSystemManager = library.EmscriptenFileSystemManager;
    if (EmscriptenFileSystemManager.isInitialized()) {
      return;
    }
    EmscriptenFileSystemManager.initializeFileSystem(fsType);
    return await waitForFileSystemInitalized();
  };
  loadHnswlib = async (inputFsType) => {
    try {
      if (typeof hnswlib !== "undefined" && hnswlib !== null) {
        const lib = hnswlib();
        if (lib != null)
          return lib;
      }
      if (!library) {
        const temp = await import("./hnswlib-317962d7.js");
        const factoryFunc = temp.default;
        library = await factoryFunc();
        await initializeFileSystemAsync(inputFsType);
        return library;
      }
      return library;
    } catch (err) {
      console.error("----------------------------------------");
      console.error("Error initializing the library:", err);
      throw err;
    }
  };
})();
export {
  IDBFS_STORE_NAME,
  __tla,
  defaultParams,
  hnswParamsForAda,
  loadHnswlib,
  syncFileSystem,
  waitForFileSystemInitalized,
  waitForFileSystemSynced
};
