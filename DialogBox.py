import easygui

def open_dialog_box(choices):
    # Open the multi-choice dialog box and store the user's selections
    selection = easygui.multchoicebox(msg='Select one or more options:', title='Multi-Choice Box', choices=choices)
    # Return the user's selections
    return selection

def open_1dialog_box(choices):
    # Show the choice box and store the user's selection
    selection = easygui.choicebox(msg='Please select one choice:', title='Choice Box', choices=choices)
    # Return the user's selections
    return selection
