*** Settings ***
Library    ../src/utils.py
Library    ../src/main.py
Library    Collections
Library    OperatingSystem


*** Variables ***
${img_path1}    ../test_images/sample1.png
${img_path2}    ../test_images/sample2.png
${pre_trained_model}    ../models/pretrained/
${fine_tuned_model}    ../models/fine_tuned/

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
    # Create Image Table
    ${table_data}    Collections.Convert To List    ${EMPTY}
    @{table_header}    Set Variable    OCropped Image    Tesseract Output    Paddle Output
    Collections.Append To List    ${table_data}    ${table_header}

    FOR    ${region}   IN    region_1    region_2    region_3    region_4    region_5
    ...    region_6    region_7    region_8    region_9
        ${cropped_image_path}=    utils.Crop Image Region    ${img_path2}   ${region}
        # ${cropped_image_path}=    utils.Enhance Image    ${cropped_image_path}

        ${tes_ocr_op}=    utils.Read Text From Image    ${cropped_image_path}
        ${paddle_ocr_op}=    main.Recognize Text    ${cropped_image_path}    ${pre_trained_model}

        ${cropped_img_html}=    utils.Log Image    img_path=${cropped_image_path}    width=200

        @{table_img_row}    Set Variable    ${cropped_img_html}    ${tes_ocr_op}    ${paddle_ocr_op}
        Collections.Append To List    ${table_data}    ${table_img_row}
    END
    Add Table To Report    @{table_data}
