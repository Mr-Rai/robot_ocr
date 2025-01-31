*** Settings ***
Library    ../src/utils.py
Library    Collections


*** Variables ***
${img_path1}    ../images/test_images/sample1.jpg
${img_path2}    ../images/test_images/sample2.png

*** Keywords ***
Add Table To Report
    [Documentation]    Accepts list of list and log it as a table in report. First row is header.
    [Arguments]    @{data}
    ${html}=    Set Variable    <table border="1" style="border-collapse: collapse;">\n
    FOR    ${row}    IN    @{data}
        ${html}=    Set Variable    ${html}<tr>
        FOR    ${cell}    IN    @{row}
            ${html}=    Set Variable    ${html}<td>${cell}</td>
        END
        ${html}=    Set Variable    ${html}</tr>\n
    END
    ${html}=    Set Variable    ${html}</table>
    Log    ${html}    html=True


*** Tasks ***
Unit Test
    [Documentation]    Unit Test For OCR Model
    # First sample
    ${cropped_image_path}=    Crop Image Region    ${img_path1}    sample1
    ${text1}    ${enhanced_image_path1}=    utils.Read Text From Image    image_path=${cropped_image_path}

    # Second sample
    ${text2}    ${enhanced_image_path2}=    utils.Read Text From Image    image_path=${img_path2}

    Log    \nText: ${text1}
    Log    \nText: ${text2}

    # Log Images To Table in Report
    ${enh_img1_html}=    utils.Log Image    img_path=${enhanced_image_path1}    width=200
    ${enh_img2_html}=    utils.Log Image    img_path=${enhanced_image_path2}    width=200
    ${org_img1_html}=    utils.Log Image    img_path=${img_path1}    width=200
    ${org_img2_html}=    utils.Log Image    img_path=${img_path2}    width=200

    ${table_data}    Collections.Convert To List    ${EMPTY}
    @{table_header}    Set Variable    Original Image    Enhanced Image
    Collections.Append To List    ${table_data}    ${table_header}
    @{table_img_row1}    Set Variable    ${org_img1_html}    ${enh_img1_html}
    @{table_img_row2}    Set Variable    ${org_img2_html}    ${enh_img2_html}
    Collections.Append To List    ${table_data}    ${table_img_row1}
    Collections.Append To List    ${table_data}    ${table_img_row2}
    Add Table To Report    @{table_data}