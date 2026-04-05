"""
Simple file picker - hide drag/drop area and show buttons to open file dialog.
"""

import streamlit as st


def simple_file_picker(uploader_key: str):
    """
    Show file uploader but hide the drag/drop UI.
    Returns the uploaded file.
    """
    # Normal file uploader (will be hidden visually)
    uploaded_file = st.file_uploader(
        "Upload",
        type=["jpg", "png", "jpeg"],
        key=uploader_key,
        label_visibility="hidden"
    )
    
    # CSS tweaks: hide dropzone and filename but keep the file input present (invisible)
    st.markdown(f"""
    <style>
    /* try to remove/hide common uploader elements */
    [data-testid="stFileUploadDropzone"],
    [data-testid^="stFileUpload"] {{
        display: none !important;
    }}

    /* hide the name of any selected file */
    div[data-testid="uploadedFileName"] {{
        display: none !important;
    }}

    /* make the actual <input> tiny and transparent but not removed from DOM
       so JS click() will work */
    input[type="file"] {{
        /* make the input off-screen but still actionable */
        position: absolute !important;
        left: -9999px !important;
        width: 0 !important;
        height: 0 !important;
        opacity: 0 !important;
        /* IMPORTANT: Do NOT set pointer-events:none - we need this clickable! */
    }}
    </style>
    """, unsafe_allow_html=True)
    
    # additional JS removes any lingering dropzone elements that slip through
    st.markdown("""
    <script>
    setTimeout(() => {
        // remove any element containing the default drag/drop text
        document.querySelectorAll('div').forEach(el => {
            if (el.textContent && el.textContent.includes('Drag and drop file here')) {
                el.remove();
            }
        });
        // also hide elements by data-testid prefix
        document.querySelectorAll('[data-testid^="stFileUpload"]').forEach(el => {
            el.style.display = 'none';
        });
    }, 50);
    </script>
    """, unsafe_allow_html=True)
    
    return uploaded_file
