
print("Only one member needs to submit on Gradescope. Please make sure to include all your group members into the group submission.")
group_num = input("Please enter your group number: ")
num_members = input("Please enter the number of group members: ")
name_list = []
for i in range(int(num_members)):
    name_list.append(input(f"Please enter the name of member {i+1}:"))
name_str = ','.join(name_list)
project_title = input("Please enter your Project Title here:")
video_link = input("Please copy and paste the link to your presentation Video here:")
github_page_link = input("Please copy and paste the github page link here:")


with open(f'final_report_submission_{group_num}.txt', 'w') as f:
    f.write(group_num+'\n')
    f.write(num_members+'\n')
    f.write(name_str)
    f.write('\n')
    f.write(project_title)
    f.write('\n')
    f.write(video_link)
    f.write('\n')
    f.write(github_page_link)
