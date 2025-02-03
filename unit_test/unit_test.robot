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
    ${cropped_image_path1}=    Crop Image Region    ${img_path1}    sample1
    ${text1}    ${enhanced_image_path1}=    utils.Read Text From Image    image_path=${cropped_image_path1}

    # Second sample
    ${text2}    ${enhanced_image_path2}=    utils.Read Text From Image    image_path=${img_path2}

    # Third Sample
    ${cropped_image_path2}=    Crop Image Region    ${img_path1}    sample2
    ${text3}    ${enhanced_image_path3}=    utils.Read Text From Image    image_path=${cropped_image_path2}

    # Fourth Sample
    ${cropped_image_path3}=    Crop Image Region    ${img_path1}    sample3
    ${text4}    ${enhanced_image_path4}=    utils.Read Text From Image    image_path=${cropped_image_path3}

    # Log Extracted Texts
    Log    Text1: ${text1}
    Log    Text2: ${text2}
    Log    Text3: ${text3}
    Log    Text4: ${text4}

    # Create html data for Logging Image Table in Report
    ${org_img1_html}=    utils.Log Image    img_path=${cropped_image_path1}    width=200
    ${enh_img1_html}=    utils.Log Image    img_path=${enhanced_image_path1}    width=200

    ${org_img2_html}=    utils.Log Image    img_path=${img_path2}    width=200
    ${enh_img2_html}=    utils.Log Image    img_path=${enhanced_image_path2}    width=200

    ${org_img3_html}=    utils.Log Image    img_path=${cropped_image_path2}    width=200
    ${enh_img3_html}=    utils.Log Image    img_path=${enhanced_image_path3}    width=200

    ${org_img4_html}=    utils.Log Image    img_path=${cropped_image_path3}    width=200
    ${enh_img4_html}=    utils.Log Image    img_path=${enhanced_image_path4}    width=200

    # Create Image Table
    ${table_data}    Collections.Convert To List    ${EMPTY}
    @{table_header}    Set Variable    Original Image    Enhanced Image
    Collections.Append To List    ${table_data}    ${table_header}

    @{table_img_row1}    Set Variable    ${org_img1_html}    ${enh_img1_html}
    @{table_img_row2}    Set Variable    ${org_img2_html}    ${enh_img2_html}
    @{table_img_row3}    Set Variable    ${org_img3_html}    ${enh_img3_html}
    @{table_img_row4}    Set Variable    ${org_img4_html}    ${enh_img4_html}

    Collections.Append To List    ${table_data}    ${table_img_row1}
    Collections.Append To List    ${table_data}    ${table_img_row2}
    Collections.Append To List    ${table_data}    ${table_img_row3}
    Collections.Append To List    ${table_data}    ${table_img_row4}

    Add Table To Report    @{table_data}