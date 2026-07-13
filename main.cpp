/*两数之和*/
#include<unordered_map>
#include<vector>
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        std::unordered_map<int,int> mapNumers;
        for(int i=0; i< nums.size();i++)
        {
            auto iNeedValue = target - nums[i];
            if(mapNumers.count(iNeedValue))
            {
                return {mapNumers[iNeedValue],i};
            }
            mapNumers[nums[i]] = i;
        }
        return {};
    }
};

 