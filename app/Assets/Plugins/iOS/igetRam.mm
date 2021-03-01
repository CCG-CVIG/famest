#import <mach/mach.h>
#import <mach/mach_host.h>

extern "C"
{
    
    const long igetRam(int what)
    {
        mach_port_t host_port;
        mach_msg_type_number_t host_size;
        vm_size_t pagesize;
        
        host_port = mach_host_self();
        host_size = sizeof(vm_statistics_data_t) / sizeof(integer_t);
        host_page_size(host_port, &pagesize);
        
        vm_statistics_data_t vm_stat;
        
        if (host_statistics(host_port, HOST_VM_INFO, (host_info_t)&vm_stat, &host_size) != KERN_SUCCESS)
            return -1;
        
        long used = (vm_stat.active_count + vm_stat.inactive_count + vm_stat.wire_count) * pagesize;
        
        long free = vm_stat.free_count * pagesize;
        
        //long total = used + free;
        
        long nearest = 256;
        long totalMemory = (NSInteger)([[NSProcessInfo processInfo] physicalMemory] );
        long rem = totalMemory % nearest;
        long tot = totalMemory - rem;
        if (rem >= nearest/2) {
            tot += 256;
        }
        
        
        
        long active = pagesize * vm_stat.active_count;
        long inactive = pagesize * vm_stat.inactive_count;
        
        long wired = pagesize * vm_stat.wire_count;
        
        if(what == 0) return totalMemory;
        if(what == 1) return free;
        if(what == 2) return used;
        
        if(what == 3) return active;
        if(what == 4) return inactive;
        if(what == 5) return wired;
        
        return -1;
    }
    
    
    
}
