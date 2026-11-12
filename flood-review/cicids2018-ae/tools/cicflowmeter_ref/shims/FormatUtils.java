package org.jnetpcap.packet.format;
public class FormatUtils {
    public static String ip(byte[] a) { return (a[0] & 255) + "." + (a[1] & 255) + "." + (a[2] & 255) + "." + (a[3] & 255); }
}
