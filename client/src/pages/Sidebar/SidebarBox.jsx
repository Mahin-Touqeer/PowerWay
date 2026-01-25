import { Link, NavLink, Outlet } from "react-router-dom";
import {
  SidebarHeader,
  SidebarProvider,
  SidebarTrigger,
} from "../../components/ui/sidebar";
import DashboardOutlinedIcon from "@mui/icons-material/DashboardOutlined";
import AccountCircleIcon from "@mui/icons-material/AccountCircle";
import SignalCellularAltOutlinedIcon from "@mui/icons-material/SignalCellularAltOutlined";
import SettingsIcon from "@mui/icons-material/Settings";
import DevicesIcon from "@mui/icons-material/Devices";
import NotificationsIcon from "@mui/icons-material/Notifications";
import SecurityIcon from "@mui/icons-material/Security";

import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "../../components/ui/sidebar";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import ColorText from "@/utils/ColorText";
// import styles from "./SidebarBox.module.css";

export default function SidebarBox() {
  return (
    <SidebarProvider>
      <Sidebar
        collapsible="icon"
        className="overflow-hidden"
        style={{ border: "none" }}
      >
        <SidebarHeader className="flex flex-row items-center  border-r-2 h-[49.6px] bg-white">
          <img src="/logo.jpeg" style={{ width: "2rem" }} />
          <h1 className="inline-block group-data-[collapsible=icon]:hidden font-bold text-l ml-1">
            WattNext
          </h1>
        </SidebarHeader>
        <SidebarContent className="bg-white border-r-2">
          {/* <SidebarGroup> */}
          {/* <SidebarGroupLabel>Dashboard</SidebarGroupLabel> */}
          {/* <SidebarGroupContent> */}
          <SidebarMenu>
            {pages.map((page) => {
              return (
                <NavLink to={page.href}>
                  <SidebarMenuItem
                    key={page.name}
                    className="w-full h-[48px] flex justify-start items-center"
                  >
                    <div
                      className="flex ml-[10px] iconlabel"
                      style={{ color: "#524f4f" }}
                    >
                      {page.icon}
                    </div>
                    <span
                      className="ml-[1rem] group-data-[collapsible=icon]:hidden label font-semibold"
                      style={{
                        color: "#524f4f",
                      }}
                    >
                      {page.name}
                    </span>
                  </SidebarMenuItem>
                </NavLink>
              );
            })}
          </SidebarMenu>
          {/* </SidebarGroupContent> */}
          {/* </SidebarGroup> */}
        </SidebarContent>
      </Sidebar>
      <main className={`w-full `}>
        <div className="border-b-2 flex justify-between items-center px-4">
          <SidebarTrigger style={{ cursor: "pointer", height: "48px" }} />
          <div className="flex font-bold">
            {/* <ColorText color="green" text="normal" />
            <ColorText color="#FEA600" text="warning" />
            <ColorText color="#FF6900" text="degraded" />
            <ColorText color="red" text="critical" /> */}
            Dashboard
          </div>
          <Avatar>
            <AvatarImage src="https://github.com/shadcn.png" />
            <AvatarFallback>CN</AvatarFallback>
          </Avatar>
        </div>

        <div className="p-12 bg-[#F6F8F7]">
          <Outlet />
        </div>
      </main>
    </SidebarProvider>
  );
}

const pages = [
  {
    name: "Overview",
    icon: <DashboardOutlinedIcon style={{ fontSize: "26px" }} />,
    href: "/home",
  },
  {
    name: "Load",
    icon: <DevicesIcon style={{ fontSize: "26px" }} />,
    href: "/load",
  },
  // {
  //   name: "Alerts/Notifications",
  //   icon: <NotificationsIcon style={{ color: "#524f4f", fontSize: "26px" }} />,
  //   href: "#",
  // },
  {
    name: "Graphs",
    icon: (
      <SignalCellularAltOutlinedIcon
        style={{ color: "#524f4f", fontSize: "26px" }}
      />
    ),
    href: "/graphs",
  },

  // {
  //   name: "Settings",
  //   icon: <SettingsIcon style={{ color: "#524f4f", fontSize: "26px" }} />,
  //   href: "#",
  // },
  // {
  //   name: "Security",
  //   icon: <SecurityIcon style={{ color: "#524f4f", fontSize: "26px" }} />,
  //   href: "#",
  // },
  // {
  //   name: "Account",
  //   icon: <AccountCircleIcon style={{ color: "#524f4f", fontSize: "26px" }} />,
  //   href: "#",
  // },
];
