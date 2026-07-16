const editor = grapesjs.init({

    container:"#gjs",

    height:"100vh",

    storageManager:{
        type:"local"
    },

    plugins:[
        "grapesjs-preset-webpage"
    ]

});


loadBlocks(editor);